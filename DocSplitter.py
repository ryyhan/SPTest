import os
from typing import List, Dict, Tuple
import fitz
import pytesseract
from PIL import Image
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
                self.reader = easyocr.Reader(['en'], gpu=False)
            except ImportError:
                print("Warning: easyocr is not installed. Falling back to tesseract.")
                self.ocr_engine = "tesseract"

        # Dictionary mapping form types to their unique titles
        self.form_identifiers = {
            "W-8BEN": "Certificate of Foreign Status of Beneficial Owner",
            "W-8BEN-E": "Certificate of Status of Beneficial Owner",
            "W-8EXP": "Certificate of Foreign Government",
            "W-8IMY": "Certificate of Foreign Intermediary",
            "W-9": "Request for Taxpayer Identification Number"
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
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            if self.ocr_engine == "easyocr":
                try:
                    import numpy as np
                    img_array = np.array(img)
                    results = self.reader.readtext(img_array)
                    text = ' '.join([res[1] for res in results])
                except Exception as e:
                    print(f"EasyOCR error: {e}. Falling back to tesseract.")
                    text = pytesseract.image_to_string(img)
            else:
                text = pytesseract.image_to_string(img)
            
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
        
        # First pass: Check for titles (Strong match) using flexible regex
        for form_type, _ in sorted_forms:
            pattern = title_patterns.get(form_type)
            if pattern and re.search(pattern, text, re.IGNORECASE):
                print(f"DEBUG: Found title pattern for {form_type}")
                return (form_type, True)
                
        # Second pass: Check for 'Form X' fallbacks (Weak match) using flexible regex
        # We limit check to first 1000 chars to avoid matching instructions
        first_1000 = text[:1000].lower()
        for form_type, _ in sorted_forms:
            # Create a flexible regex from the form type (e.g. "W-8BEN" -> r'w\s*[-]?\s*8\s*b\s*e\s*n')
            # This handles missing dashes, extra spaces, etc.
            clean_form = form_type.replace('-', '')
            regex_str = r'\s*[-]?\s*'.join(list(clean_form))
            
            # Look for "form [form_type]"
            if re.search(rf'form\s*[-:\n]?\s*{regex_str}\b', first_1000, re.IGNORECASE):
                print(f"DEBUG: Found 'form {form_type}' fallback via flexible regex")
                return (form_type, False)
            
            # Special fallback just looking for the form name (e.g. "W-9") 
            # We enforce word boundaries so "W-9" doesn't match inside a larger word
            if re.search(rf'\b{regex_str}\b', first_1000, re.IGNORECASE):
                print(f"DEBUG: Found '{form_type}' standalone via flexible regex")
                return (form_type, False)
                
        # Third pass: Fuzzy Matching (Safety net for very bad OCR)
        # We check if the text contains anything remotely similar to the form titles
        for form_type, title in sorted_forms:
            # partial_ratio checks if the shorter string (title) is similar to any 
            # substring of the longer string (text).
            score = fuzz.partial_ratio(title.lower(), first_1000)
            if score > 85: # 85% similarity threshold
                print(f"DEBUG: Found title via fuzzy matching (Score: {score}) for {form_type}")
                return (form_type, True)
                
            # Let's also fuzzy match the form name itself (e.g. "Form W-9")
            # We don't want to fuzzy match just "W-9", because "W-3" is 66% similar. 
            # It needs to be "Form W-9" to give it enough length for a meaningful partial_ratio.
            form_string = f"form {form_type.lower()}"
            score_form = fuzz.partial_ratio(form_string, first_1000)
            if score_form > 90: # Higher threshold for short strings
                print(f"DEBUG: Found 'form {form_type}' via fuzzy matching (Score: {score_form})")
                return (form_type, False)
                
        # Check for certificates
        if self.certificate_pattern.search(text):
            return ("CERTIFICATE", True)
            
        return ("OTHER", True)

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
        """Group pages into separate documents."""
        documents = []
        current_doc = None
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = self.extract_text_from_page(page)
            form_type, is_start_page = self.identify_form_type(text)
            page_num_in_doc = self.extract_page_number(text)
            print(f"DEBUG: Page {page_num} -> Type: {form_type}, IsStart: {is_start_page}, PageNum: {page_num_in_doc}")
            
            # Determine if we should start a new document
            start_new = False
            
            if current_doc is None:
                start_new = True
            else:
                # Check strict page count rules
                current_type = current_doc['type']
                current_len = len(current_doc['pages'])
                
                # Default behavior (Priority Logic)
                if is_start_page and form_type != "OTHER":
                    # PRIORITY 1: Explicit Title Found -> START NEW
                    start_new = True
                elif page_num_in_doc is not None and page_num_in_doc > 1:
                    # PRIORITY 2: Page > 1 -> CONTINUE
                    start_new = False
                elif page_num_in_doc == 1:
                    # PRIORITY 3: Explicit Page 1 -> START NEW
                    start_new = True
                else:
                    # PRIORITY 4: Fallback
                    if form_type == current_type:
                         # Same type, no title -> Continuation
                         start_new = False
                    elif form_type != "OTHER":
                        # Different type -> Split
                        start_new = True
                    else:
                        # OTHER -> Continuation
                        start_new = False

                # OVERRIDE WITH STRICT PAGE COUNTS
                if current_type in self.form_rules:
                    allowed_counts = self.form_rules[current_type]
                    max_count = max(allowed_counts)
                    
                    if current_len < max_count:
                        # We haven't reached the max length yet.
                        # BUT, if we are at a valid stopping point (e.g. W-9 page 1), 
                        # we need to be careful.
                        
                        if current_len in allowed_counts:
                            # We are at a valid length (e.g. W-9 Page 1).
                            # If the standard logic said "Start New" (e.g. found new title), we allow it.
                            # If standard logic said "Continue" (e.g. Page 2), we allow it.
                            pass 
                        else:
                            # We are NOT at a valid length (e.g. W-8BEN-E Page 3).
                            # We MUST continue, UNLESS we see a totally different form title 
                            # (which would mean the file is malformed/mixed up, but we should trust the title).
                            # However, user said "W-8BEN-E will have 8 pages".
                            # So we should bias heavily towards continuation.
                            
                            if start_new and (is_start_page and form_type != "OTHER"):
                                # We found a NEW form title. This contradicts the page count rule.
                                # E.g. We are on Page 3 of W-8BEN-E, but found "Form W-9".
                                # We should probably respect the Title (file might be truncated).
                                pass
                            else:
                                # Otherwise, force continuation
                                start_new = False
                    
                    elif current_len >= max_count:
                        # We reached max length. Force split for next page.
                        start_new = True
            
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