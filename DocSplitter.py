import os
import sys
from typing import List, Dict, Tuple
import fitz
import pytesseract
from PIL import Image
try:
    import cv2
    cv2.setNumThreads(4) # Force OpenCV to use multiple CPU cores for preprocessing
    cv2.setUseOptimized(True) # Force SIMD/Intel optimizations
except ImportError:
    pass
import re
from collections import defaultdict
from thefuzz import fuzz

class PDFSplitter:
    def __init__(self, ocr_engine="tesseract"):
        self.ocr_engine = ocr_engine
        
        if self.ocr_engine == "easyocr":
            try:
                import easyocr
                # Initialize the EasyOCR reader once to avoid reloading model per page
                # Set gpu=True if you have a GPU available
                # Set quantize=False as native floating-point math is often faster on modern CPUs than quantized integers
                self.reader = easyocr.Reader(['en'], gpu=False, quantize=False)
            except ImportError:
                print("Warning: easyocr is not installed. Falling back to tesseract.")
                self.ocr_engine = "tesseract"
                
        elif self.ocr_engine == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.reader = RapidOCR()
            except ImportError:
                print("Warning: rapidocr_onnxruntime is not installed. Falling back to tesseract.")
                self.ocr_engine = "tesseract"

        # Dictionary mapping form types to their unique titles
        self.form_identifiers = {
            "W-8BEN": "Certificate of Foreign Status of Beneficial Owner",
            "W-8BEN-E": "Certificate of Status of Beneficial Owner",
            "W-8EXP": "Certificate of Foreign Government",
            "W-8IMY": "Certificate of Foreign Intermediary",
            "W-9": "Request for Taxpayer Identification Number"
        }
        
        # Unique catalog numbers and keywords for disambiguation
        self.unique_identifiers = {
            "W-8BEN": [r'\b25047z\b'],
            "W-8BEN-E": [r'\b59689n\b', r'\bnonparticipating\b'],
            "W-8EXP": [r'\b115\(2\)\b', r'\b1443\(b\)\b', r'\b897\(l\)-1\(d\)\b'],
            "W-8IMY": [r'\b25402q\b', r'\bqi-ein\b', r'\bwp-ein\b', r'\bwt-ein\b'],
            "W-9": [r'\b10231x\b', r'\b1099-int\b', r'\b1099-misc\b']
        }
        
        # Pattern to identify certificates
        self.certificate_pattern = re.compile(r'certificate', re.IGNORECASE)
        
        # Strict page count rules
        self.form_rules = {
            "W-9": [1, 6],
            "W-8BEN": [1],
            "W-8BEN-E": [8],
            "W-8EXP": [3],
            "W-8IMY": [8]
        }
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        # Remove extra whitespace and newlines
        return ' '.join(text.replace('\n', ' ').split())

    def extract_text_from_page(self, page) -> str:
        """Extract text from a page using both native text extraction and OCR if needed."""
        # Try native text extraction first
        text = page.get_text()
        
        # If very little text is found (likely scanned or noise), try OCR
        if len(text.strip()) < 50:
            pix = page.get_pixmap()
            
            try:
                import cv2
                import numpy as np
                
                # 1. Convert PyMuPDF pixmap to numpy array
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                # 2. Convert to Grayscale
                # PyMuPDF typically returns RGB or RGBA depending on the PDF internal format
                if pix.n >= 3:
                     gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                     gray = img_array # Already grayscale
                     
                # 3. Apply Otsu's thresholding
                # This mathematically finds the perfect divide between "ink" and "paper", forcing everything to pure black/white
                # It completely removes shadows, weird lighting gradients, and compression artifacts
                # Note: Deep learning models (EasyOCR, RapidOCR) prefer grayscale (anti-aliased) text, 
                # so we only apply hard thresholding for Tesseract
                if self.ocr_engine in ["easyocr", "rapidocr"]:
                     processed_img = gray
                else:
                     _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            except ImportError:
                print("Warning: cv2 or numpy not installed. Skipping image preprocessing.")
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                processed_img = np.array(img) if 'numpy' in sys.modules else img
                
            
            if self.ocr_engine == "rapidocr":
                try:
                    # RapidOCR takes numpy arrays directly
                    result, _ = self.reader(processed_img)
                    if result:
                        text = ' '.join([res[1] for res in result])
                    else:
                        text = ""
                except Exception as e:
                    print(f"RapidOCR error: {e}. Falling back to tesseract.")
                    pil_img = Image.fromarray(processed_img)
                    text = pytesseract.image_to_string(pil_img)
                    
            elif self.ocr_engine == "easyocr":
                try:
                    # EasyOCR takes numpy arrays directly. We use optimal params for reading dense documents.
                    results = self.reader.readtext(
                        processed_img,
                        paragraph=True,           # Combine words into paragraphs
                        x_ths=0.8,                # Tolerate more horizontal distance between words
                        y_ths=0.5,                # Tolerate slight vertical shifts
                        text_threshold=0.5,       # Be more forgiving on low-confidence letters
                    )
                    text = ' '.join([res[1] for res in results])
                except Exception as e:
                    print(f"EasyOCR error: {e}. Falling back to tesseract.")
                    # Tesseract prefers PIL Images, so we convert back
                    pil_img = Image.fromarray(processed_img)
                    text = pytesseract.image_to_string(pil_img)
            else:
                # Tesseract prefers PIL Images, so we convert back
                try:
                    pil_img = Image.fromarray(processed_img)
                except Exception:
                    # Fallback if preprocessing failed (e.g. cv2 not installed and img is a PIL Object)
                    pil_img = processed_img if isinstance(processed_img, Image.Image) else Image.fromarray(processed_img)
                    
                # We can also add a PSM flag here for standard forms: block of text
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(pil_img, config=custom_config)
            
        return self.clean_text(text)

    def identify_form_type(self, text: str) -> Tuple[str, bool]:
        """
        Identify the type of form based on text content using flexible regex.
        Returns a tuple of (form_type, is_start_page).
        """
        # Dictionary of regex patterns for titles
        title_patterns = {
            "W-8BEN": r'certificate.*?foreign\s*status.*?beneficial\s*owner',
            "W-8BEN-E": r'certificate.*?status.*?beneficial\s*owner.*?entities',
            "W-8EXP": r'certificate.*?foreign\s*government',
            "W-8IMY": r'certificate.*?foreign\s*intermediary',
            "W-9": r'request.*?taxpayer\s*identification\s*number'
        }
        
        # Sort by length of form key to check specific forms first (e.g. W-8BEN-E before W-8BEN)
        sorted_forms = sorted(self.form_identifiers.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Pass 0: Check for highly specific unique identifiers (Catalog numbers, unique EIN refs)
        # These are practically impossible to false-match.
        text_lower = text.lower()
        forced_form_type = None
        for form_type, patterns in self.unique_identifiers.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    print(f"DEBUG: Found unique identifier '{pattern}' for {form_type}. Forcing form type.")
                    forced_form_type = form_type
                    break
            if forced_form_type:
                break
        
        # Helper to return the correct form type
        def get_return_val(matched_type, is_start):
            final_type = forced_form_type if forced_form_type else matched_type
            if final_type != matched_type and forced_form_type:
                 print(f"DEBUG: Overriding matched type '{matched_type}' with forced type '{final_type}'")
            return (final_type, is_start)
        
        # First pass: Check for titles (Strong match) using flexible regex
        earliest_title_type = None
        earliest_title_idx = float('inf')
        
        for form_type, _ in sorted_forms:
            pattern = title_patterns.get(form_type)
            if pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.start() < earliest_title_idx:
                    earliest_title_idx = match.start()
                    earliest_title_type = form_type
                    
        if earliest_title_type:
            print(f"DEBUG: Found title pattern for {earliest_title_type} at idx {earliest_title_idx}")
            return get_return_val(earliest_title_type, True)
                
        # Second pass: Check for 'Form X' fallbacks (Weak match) using flexible regex
        # We limit check to first 1000 chars to avoid matching instructions
        first_1000 = text[:1000].lower()
        
        # To avoid misclassifying Forms that mention OTHER forms in their instructions (e.g. W-8IMY mentions W-8BEN-E),
        # we will collect all matches and pick the one that appears *earliest* in the text.
        earliest_match_type = None
        earliest_match_idx = float('inf')
        
        for form_type, _ in sorted_forms:
            clean_form = form_type.replace('-', '')
            regex_str = r'\s*[-]?\s*'.join(list(clean_form))
            
            # Look for "form [form_type]"
            match_form = re.search(rf'form\s*[-:\n]?\s*{regex_str}\b', first_1000, re.IGNORECASE)
            if match_form:
                if match_form.start() < earliest_match_idx:
                    earliest_match_idx = match_form.start()
                    earliest_match_type = form_type
            
            # Special fallback just looking for the form name (e.g. "W-9") 
            match_standalone = re.search(rf'\b{regex_str}\b', first_1000, re.IGNORECASE)
            if match_standalone:
                if match_standalone.start() < earliest_match_idx:
                    earliest_match_idx = match_standalone.start()
                    earliest_match_type = form_type
                    
        if earliest_match_type:
            # If the mention is very early in the text (first 300 chars), 
            # it's almost certainly the actual form header (top-left of doc), not instructions.
            is_start = earliest_match_idx < 300
            print(f"DEBUG: Found '{earliest_match_type}' as the earliest form mention via flexible regex (idx: {earliest_match_idx}, is_start: {is_start})")
            return get_return_val(earliest_match_type, is_start)
                
                
        # Third pass: Fuzzy Matching (Safety net for very bad OCR)
        # We check if the text contains anything remotely similar to the form titles
        earliest_fuzzy_type = None
        earliest_fuzzy_idx = float('inf')
        is_fuzzy_title_match = False
        
        for form_type, title in sorted_forms:
            # We want to find *where* the fuzzy match occurred to see if it's the earliest
            # Thefuzz doesn't give us the index easily, so we use regex to find the string that fuzz matched
            
            # Fuzzy match the form name itself (e.g. "Form W-9")
            form_string = f"form {form_type.lower()}"
            score_form = fuzz.partial_ratio(form_string, first_1000)
            
            if score_form > 90: # Higher threshold for short strings
                # Try to find exactly where this matched in the text so we can prioritize the earliest
                clean_form = form_type.replace('-', '')
                regex_str = r'\s*[-]?\s*'.join(list(clean_form))
                match = re.search(rf'form\s*[-:\n]?\s*{regex_str}\b', first_1000, re.IGNORECASE)
                
                # If we found it via regex (which means we got an index), use that index
                # If not, let's just assume it's valid and if it's the only one, we use it.
                idx = match.start() if match else float('inf')
                
                if idx < earliest_fuzzy_idx:
                    earliest_fuzzy_idx = idx
                    earliest_fuzzy_type = form_type
                    is_fuzzy_title_match = False
                    
            # Let's also check the long title
            score_title = fuzz.partial_ratio(title.lower(), first_1000)
            if score_title > 85:
                # If a long title fuzzy matches, it's extremely strong evidence. 
                # Titles are usually at the very top.
                print(f"DEBUG: Found title via fuzzy matching (Score: {score_title}) for {form_type}")
                return get_return_val(form_type, True)
                
        if earliest_fuzzy_type:
            # If the mention is very early in the text (first 300 chars),
            # it's almost certainly the actual form header, not instructions.
            is_start = is_fuzzy_title_match or (earliest_fuzzy_idx < 300)
            print(f"DEBUG: Found '{earliest_fuzzy_type}' as the earliest form mention via fuzzy matching (idx: {earliest_fuzzy_idx}, is_start: {is_start})")
            return get_return_val(earliest_fuzzy_type, is_start)
                
        # Check for certificates
        if self.certificate_pattern.search(text):
            return get_return_val("CERTIFICATE", True)
            
        return get_return_val("OTHER", True)

    def extract_page_number(self, text: str) -> int:
        """Extract page number from text if present."""
        # Look for "Page X" or "Page X of Y"
        # Make regex stricter to avoid "see page 2"
        # We look for "Page X" at the start of a line or after a newline, 
        # or "Page X of Y"
        # Also handle "Page: X"
        
        # Pattern 1: "Page 1 of 6"
        match = re.search(r'page\s+(\d+)\s+of\s+\d+', text.lower())
        if match:
            return int(match.group(1))
            
        # Pattern 3: W-9 specific footer like "Form W-9 (Rev. 10-2018) Page X"
        match = re.search(r'form w-9.*?page\s+(\d+)', text.lower())
        if match:
            return int(match.group(1))
            
        # Let's try to match "Page X" at the beginning of a line or end
        matches = re.findall(r'page\s+(\d+)', text.lower())
        if matches:
            # Usually the page number is the largest number? Or the one that matches sequence?
            # For now, let's take the last one found (often footer)
            return int(matches[-1])
            
        return None

    def group_pages(self, pdf_document) -> List[Dict]:
        """Group pages into separate documents strictly by physical page count."""
        documents = []
        current_doc = None
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = self.extract_text_from_page(page)
            form_type, is_start_page = self.identify_form_type(text)
            print(f"DEBUG: Page {page_num} -> Type: {form_type}, IsStart: {is_start_page}")
            
            start_new = False
            
            if current_doc is None:
                start_new = True
            else:
                current_type = current_doc['type']
                current_len = len(current_doc['pages'])
                
                # Check if we are currently building a form with known strict length rules
                if current_type in self.form_rules:
                    allowed_counts = self.form_rules[current_type]  # e.g., [1, 6] for W-9, or [8] for W-8IMY
                    max_count = max(allowed_counts)
                    
                    if current_len < max_count:
                        # We have NOT reached the maximum allowed length for this form yet.
                        # Rule 1: Always absorb the page, UNLESS we are at a valid stopping point
                        # AND the new page is definitively starting a new form.
                        if current_len in allowed_counts and is_start_page and form_type != "OTHER" and form_type != current_type:
                            start_new = True
                        else:
                            # Blindly absorb! Ignore any text, footers, or fake titles.
                            # If it's page 2 of a 104-page PDF, and page 1 was W-8IMY, 
                            # page 2 belongs to W-8IMY regardless of what's printed on it.
                            start_new = False
                    else:
                        # We have hit the absolute maximum length for this form.
                        # The very next page MUST start a new document.
                        start_new = True
                
                else:
                    # We are building a form with NO strict length rules (e.g. "OTHER" or "CERTIFICATE")
                    # We only split when the scanner detects a brand new, definitive form title.
                    if is_start_page and form_type != "OTHER" and form_type != current_type:
                        start_new = True
                    else:
                        start_new = False
            
            if start_new:
                if current_doc is not None:
                    documents.append(current_doc)
                
                # Create new document
                doc_id = len([d for d in documents if d['type'] == form_type]) + 1
                current_doc = {
                    'type': form_type,
                    'pages': [],
                    'id': doc_id,
                    'text': text  # Store first page text for naming certificates
                }
            
            current_doc['pages'].append(page_num)
            
        # Add the last document
        if current_doc is not None:
            documents.append(current_doc)
            
        return documents

    def generate_filename(self, doc: Dict) -> str:
        """Generate appropriate filename for the document."""
        if doc['type'] == "CERTIFICATE":
            # Extract a meaningful name from the certificate text
            # Find first instance of "certificate" and take surrounding words
            text = doc['text'].lower()
            cert_idx = text.find("certificate")
            if cert_idx != -1:
                # Take up to 5 words before and after "certificate"
                words = text.split()
                cert_word_idx = next(i for i, word in enumerate(words) if "certificate" in word.lower())
                start_idx = max(0, cert_word_idx - 5)
                end_idx = min(len(words), cert_word_idx + 6)
                cert_name = "_".join(words[start_idx:end_idx])
                # Clean the filename
                cert_name = re.sub(r'[^\w\-_.]', '_', cert_name)
                return f"certificate_{doc['id']}_{cert_name[:50]}.pdf"
        elif doc['type'] == "OTHER":
            return f"other_document_{doc['id']}.pdf"
        else:
            return f"{doc['type'].lower()}_{doc['id']}.pdf"

    def split_pdf(self, input_path: str, output_dir: str) -> List[Dict]:
        """Split PDF into separate files based on form types."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the PDF
        pdf_document = fitz.open(input_path)
        
        # Group pages into separate documents
        documents = self.group_pages(pdf_document)
        
        # Create separate PDFs for each document
        for doc in documents:
            new_pdf = fitz.open()
            for page_num in doc['pages']:
                new_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
            
            output_filename = self.generate_filename(doc)
            output_path = os.path.join(output_dir, output_filename)
            new_pdf.save(output_path)
            new_pdf.close()
        
        pdf_document.close()
        return documents

def main():
    # You can change to "easyocr" to test locally
    splitter = PDFSplitter(ocr_engine="tesseract") 
    input_pdf = "ooo.pdf"
    output_directory = "split_forms"
    
    try:
        result = splitter.split_pdf(input_pdf, output_directory)
        
        # Print summary
        print("\nPDF splitting complete! Summary:")
        for doc in result:
            print(f"{doc['type']} (ID: {doc['id']}): {len(doc['pages'])} page(s)")
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()