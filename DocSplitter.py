import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
import pytesseract
from PIL import Image
try:
    import cv2
    import numpy as np
    cv2.setNumThreads(4) # Force OpenCV to use multiple CPU cores for preprocessing
    cv2.setUseOptimized(True) # Force SIMD/Intel optimizations
except ImportError:
    cv2 = None
    numpy = None
import re
from collections import defaultdict
from thefuzz import fuzz

# Optional LLM support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


@dataclass
class PageAnalysis:
    """Stores analysis results for a single page."""
    page_num: int
    text: str
    form_type: str
    is_start_page: bool
    confidence: float
    has_images: bool
    text_length: int
    matched_patterns: List[str]
    is_ambiguous: bool = False
    ambiguous_forms: List[str] = None

class PDFSplitter:
    # Confidence thresholds for different detection tiers
    CONFIDENCE_UNIQUE_ID = 100.0   # Unique catalog number match
    CONFIDENCE_TITLE = 95.0        # Full title pattern match
    CONFIDENCE_FORM = 80.0         # Form name match
    CONFIDENCE_FUZZY = 60.0        # Fuzzy match
    CONFIDENCE_CERTIFICATE = 70.0  # Certificate pattern
    
    # OCR threshold settings
    OCR_TEXT_LENGTH_THRESHOLD = 100  # Trigger OCR if text < this
    OCR_IMAGE_COUNT_THRESHOLD = 1    # Trigger OCR if page has images AND low text
    
    # Common OCR confusions for generating variants
    OCR_CONFUSIONS = {
        '0': ['O', 'o'],
        'O': ['0', 'o'],
        'o': ['0', 'O'],
        '1': ['l', 'I', 'i', '|'],
        'l': ['1', 'I', 'i'],
        'I': ['1', 'l', 'i', '|'],
        'i': ['1', 'l', 'I'],
        '5': ['S', 's'],
        'S': ['5', 's'],
        's': ['5', 'S'],
        '8': ['B', 'b'],
        'B': ['8', 'b'],
        'b': ['8', 'B'],
        'Z': ['2', 'z'],
        '2': ['Z', 'z'],
        'z': ['Z', '2'],
        '6': ['G', 'g'],
        'G': ['6', 'g'],
        '9': ['g', 'q'],
        'g': ['9', 'q'],
        'm': ['rn', 'rn'],
        'w': ['vv', 'VV'],
    }
    
    # LLM Configuration
    LLM_FORM_SYSTEM_PROMPT = """You are an expert document classification assistant specialized in US tax forms. Your task is to analyze text extracted from PDF pages and accurately identify the form type.

AVAILABLE FORM TYPES (in priority order):
1. W-8BEN: "Certificate of Foreign Status of Beneficial Owner for United States Tax Withholding and Reporting (Individuals)"
   - Key identifiers: Catalog number 25047Z, "individuals", "beneficial owner", "foreign status"
   - Typical length: 1 page

2. W-8BEN-E: "Certificate of Status of Beneficial Owner for United States Tax Withholding and Reporting (Entities)"
   - Key identifiers: Catalog number 59689N, "entities", "beneficial owner", "nonparticipating"
   - Typical length: 8 pages
   - IMPORTANT: Distinguish from W-8BEN by looking for "entities" vs "individuals"

3. W-8EXP: "Certificate of Foreign Government or Other Foreign Organization for United States Tax Withholding and Reporting"
   - Key identifiers: Catalog numbers 115(2), 1443(b), 897(l)-1(d), "foreign government"
   - Typical length: 3 pages

4. W-8IMY: "Certificate of Foreign Intermediary, Foreign Flow-Through Entity, or Certain U.S. Branches for United States Tax Withholding and Reporting"
   - Key identifiers: Catalog number 25402Q, "intermediary", "qi-ein", "wp-ein", "wt-ein"
   - Typical length: 8 pages

5. W-9: "Request for Taxpayer Identification Number and Certification"
   - Key identifiers: Catalog number 10231X, "1099-INT", "1099-MISC", "TIN", "EIN"
   - Typical length: 1 or 6 pages (with instructions)

6. CERTIFICATE: Any certificate document that doesn't match the above forms
   - Look for words like "certificate", "certification", "certified"

7. OTHER: Any document that doesn't match the above categories

CLASSIFICATION RULES:
1. Catalog numbers are the STRONGEST signal - if you see one, trust it above all else
2. Form titles in headers are very strong signals
3. Be careful of form instructions that mention other form types (e.g., W-8IMY instructions mention W-8BEN-E)
4. Consider the overall document structure and layout
5. OCR text may contain errors - use context to infer correct values (e.g., "25O47Z" is likely "25047Z")
6. If the page contains multiple form references, identify which form this page BELONGS TO (not just mentions)

You will receive text extracted from a single page. Return your classification as JSON."""

    LLM_FORM_USER_TEMPLATE = """Analyze this text from page {page_num} of a PDF and classify the form type.

TEXT EXTRACT (may contain OCR errors):
---
{text}
---

Respond with ONLY a valid JSON object in this exact format:
{{
    "form_type": "W-8BEN",
    "confidence": 0.95,
    "reasoning": "Brief explanation of your decision",
    "is_first_page": true,
    "detected_catalog": "25047Z or null if not found",
    "alternative_forms": ["W-8BEN-E"] if any, otherwise omit
}}

form_type must be one of: W-8BEN, W-8BEN-E, W-8EXP, W-8IMY, W-9, CERTIFICATE, OTHER
confidence should be between 0.0 and 1.0
is_first_page should be true if this appears to be the first page of a multi-page form
"""
    
    def __init__(self, ocr_engine="tesseract", use_llm=False, api_key=None, api_base=None, llm_model="gpt-4o-mini"):
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
        
        # Pre-compute OCR variants for all unique identifiers (cached for performance)
        self._catalog_variants_cache = {}
        self._build_catalog_variants_cache()
        
        # LLM Configuration
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.api_key = api_key
        self.api_base = api_base
        self.client = None
        
        if self.use_llm:
            if not OPENAI_AVAILABLE:
                print("Warning: openai package not installed. LLM classification disabled.")
                self.use_llm = False
            elif not api_key:
                print("Warning: No API key provided. LLM classification disabled.")
                self.use_llm = False
            else:
                try:
                    if api_base:
                        self.client = OpenAI(api_key=api_key, base_url=api_base)
                    else:
                        self.client = OpenAI(api_key=api_key)
                    print(f"LLM classification enabled using model: {llm_model}")
                except Exception as e:
                    print(f"Warning: Failed to initialize OpenAI client: {e}. LLM classification disabled.")
                    self.use_llm = False
    
    def _build_catalog_variants_cache(self):
        """Pre-compute OCR variants for all catalog numbers."""
        for form_type, patterns in self.unique_identifiers.items():
            self._catalog_variants_cache[form_type] = []
            for pattern in patterns:
                # Remove regex word boundaries for variant generation
                catalog = pattern.strip(r'\b')
                # Generate variants only for alphanumeric catalogs (not regex patterns with parens)
                if not any(c in catalog for c in ['(', ')', '[', ']', '-']):
                    variants = self.generate_ocr_variants(catalog)
                    self._catalog_variants_cache[form_type].extend(variants)
    
    def generate_ocr_variants(self, text: str, max_variants: int = 50) -> List[str]:
        """
        Generate common OCR misreadings for a given text.
        
        Args:
            text: The original text (e.g., "25047z")
            max_variants: Maximum number of variants to generate
        
        Returns:
            List of variant strings that might appear in OCR output
        """
        variants = set()
        variants.add(text.lower())  # Always include original (lowercase)
        
        # Single character substitutions (most common OCR errors)
        for i, char in enumerate(text):
            if char in self.OCR_CONFUSIONS:
                for replacement in self.OCR_CONFUSIONS[char]:
                    variant = text[:i] + replacement + text[i+1:]
                    variants.add(variant.lower())
        
        # Double character substitutions (less common but still frequent)
        if len(text) >= 2:
            for i in range(len(text) - 1):
                char1, char2 = text[i], text[i+1]
                if char1 in self.OCR_CONFUSIONS and char2 in self.OCR_CONFUSIONS:
                    for rep1 in self.OCR_CONFUSIONS[char1][:2]:  # Limit to avoid explosion
                        for rep2 in self.OCR_CONFUSIONS[char2][:2]:
                            variant = text[:i] + rep1 + rep2 + text[i+2:]
                            variants.add(variant.lower())
        
        # Common multi-char confusions
        multi_char_subs = {
            'rn': 'm',
            'm': 'rn',
            'vv': 'w',
            'VV': 'w',
            'cl': 'd',
            'd': 'cl',
        }
        for orig, sub in multi_char_subs.items():
            if orig in text:
                variants.add(text.replace(orig, sub).lower())
            if sub in text:
                variants.add(text.replace(sub, orig).lower())
        
        return list(variants)[:max_variants]
    
    def fuzzy_match_catalog(self, text: str, expected_catalog: str) -> Tuple[bool, float, str]:
        """
        Check if text contains a catalog number with fuzzy matching.
        
        Args:
            text: The OCR-extracted text to search
            expected_catalog: The expected catalog number (e.g., "25047z")
        
        Returns:
            Tuple of (found: bool, confidence: float, matched_variant: str)
        """
        text_lower = text.lower()
        expected_lower = expected_catalog.lower()
        
        # Level 1: Exact match (highest confidence)
        if expected_lower in text_lower:
            return True, 100.0, expected_lower
        
        # Level 2: Check pre-computed OCR variants (high confidence)
        if expected_catalog not in self._catalog_variants_cache:
            # Cache miss - generate variants now
            variants = self.generate_ocr_variants(expected_catalog)
            self._catalog_variants_cache[expected_catalog] = variants
        else:
            variants = self._catalog_variants_cache[expected_catalog]
        
        for variant in variants:
            if variant in text_lower:
                return True, 95.0, variant
        
        # Level 3: Fuzzy string matching (medium confidence)
        # Use thefuzz to find partial matches
        # Slide a window of catalog length across the text
        catalog_len = len(expected_catalog)
        best_score = 0
        best_match = ""
        
        for i in range(len(text_lower) - catalog_len + 1):
            window = text_lower[i:i + catalog_len]
            score = fuzz.ratio(window, expected_lower)
            if score > best_score:
                best_score = score
                best_match = window
        
        # Also check partial_ratio which handles substring matching better
        partial_score = fuzz.partial_ratio(expected_lower, text_lower)
        
        best_score = max(best_score, partial_score)
        
        if best_score >= 75:  # 75% similarity threshold
            return True, best_score, best_match
        
        # Level 4: Check for character-level similarity with position tolerance
        # This handles cases where OCR adds/removes characters
        if self._levenshtein_contains(text_lower, expected_lower, max_distance=2):
            return True, 70.0, "levenshtein_match"
        
        return False, 0.0, ""
    
    def _levenshtein_contains(self, text: str, pattern: str, max_distance: int = 2) -> bool:
        """
        Check if text contains pattern with at most max_distance edits.
        Handles OCR errors that add/remove characters.
        """
        if len(pattern) < 3:
            return False  # Too short for meaningful fuzzy match
        
        # Slide window and check Levenshtein distance
        for i in range(len(text) - len(pattern) + max_distance + 1):
            for window_size in range(len(pattern) - max_distance, len(pattern) + max_distance + 1):
                if i + window_size > len(text):
                    continue
                window = text[i:i + window_size]
                distance = self._levenshtein_distance(window, pattern)
                if distance <= max_distance:
                    return True
        return False
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def classify_page_with_llm(self, text: str, page_num: int) -> Optional[Dict]:
        """
        Use LLM to classify a page based on its text content.
        
        Args:
            text: Extracted text from the page
            page_num: Page number (0-indexed)
        
        Returns:
            Dictionary with classification results, or None if LLM fails
        """
        if not self.use_llm or not self.client:
            return None
        
        try:
            # Truncate text to avoid token limits (leave room for prompt + response)
            # GPT-4o-mini has 128K context, but we'll be conservative
            max_text_length = 10000
            if len(text) > max_text_length:
                # Keep beginning and end, which usually has the most important info
                text = text[:max_text_length // 2] + "\n...[truncated]...\n" + text[-max_text_length // 2:]
            
            # Build the prompt
            user_prompt = self.LLM_FORM_USER_TEMPLATE.format(
                page_num=page_num + 1,  # 1-indexed for LLM
                text=text
            )
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": self.LLM_FORM_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=400
            )
            
            # Parse the response
            import json
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            # Handle cases where LLM adds markdown code blocks
            if response_text.startswith("```"):
                # Remove markdown code block if present
                lines = response_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines)
            
            # Find JSON in response (in case there's extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            result = json.loads(response_text)
            
            # Map LLM response to our format
            form_type = result.get("form_type", "OTHER").upper()
            confidence = float(result.get("confidence", 0.5)) * 100
            reasoning = result.get("reasoning", "LLM classification")
            is_first_page = result.get("is_first_page", False)
            detected_catalog = result.get("detected_catalog")
            alternative_forms = result.get("alternative_forms", [])
            
            # Validate form type
            valid_types = ["W-8BEN", "W-8BEN-E", "W-8EXP", "W-8IMY", "W-9", "CERTIFICATE", "OTHER"]
            if form_type not in valid_types:
                print(f"Warning: LLM returned invalid form_type '{form_type}', defaulting to OTHER")
                form_type = "OTHER"
            
            # Build patterns list
            patterns = [f"llm:{form_type}"]
            if detected_catalog:
                patterns.append(f"llm_catalog:{detected_catalog}")
            
            # Check for ambiguity
            is_ambiguous = len(alternative_forms) > 0
            
            return {
                'form_type': form_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'is_start_page': is_first_page,
                'matched_patterns': patterns,
                'is_ambiguous': is_ambiguous,
                'ambiguous_forms': alternative_forms,
                'detected_catalog': detected_catalog,
                'method': 'llm'
            }
            
        except json.JSONDecodeError as e:
            print(f"LLM response parsing error: {e}. Response: {response_text}")
            return None
        except Exception as e:
            print(f"LLM classification error: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        # Remove extra whitespace and newlines
        return ' '.join(text.replace('\n', ' ').split())

    def should_trigger_ocr(self, text: str, page) -> Tuple[bool, dict]:
        """
        Determine if OCR should be triggered based on multiple factors.
        Returns (should_ocr, metadata_dict).
        """
        metadata = {
            'text_length': len(text.strip()),
            'has_images': False,
            'reason': 'native_text_sufficient'
        }
        
        # Check for images on the page
        try:
            images = page.get_images()
            metadata['has_images'] = len(images) > 0
        except Exception:
            pass
        
        text_length = len(text.strip())
        
        # Rule 1: Very little text found - likely scanned
        if text_length < self.OCR_TEXT_LENGTH_THRESHOLD:
            if metadata['has_images']:
                metadata['reason'] = 'low_text_with_images'
            else:
                metadata['reason'] = 'low_text_no_images'
            return True, metadata
        
        # Rule 2: Text exists but page has images - might be form with signature/image
        # Only trigger if text is relatively low (< 300 chars) AND has images
        if text_length < 300 and metadata['has_images']:
            metadata['reason'] = 'moderate_text_with_images'
            return True, metadata
        
        # Rule 3: Check for "scanned" artifacts (e.g., text appears as single long line)
        lines = text.split('\n')
        if len(lines) == 1 and len(text) > 500:
            # Single long line might indicate poor PDF text extraction
            metadata['reason'] = 'single_line_text'
            return True, metadata
        
        return False, metadata

    def extract_text_from_page(self, page) -> Tuple[str, dict]:
        """
        Extract text from a page using both native text extraction and OCR if needed.
        Returns (text, metadata_dict).
        """
        # Try native text extraction first
        text = page.get_text()
        ocr_metadata = {'ocr_used': False, 'ocr_reason': None}
        
        # Determine if OCR should be triggered
        should_ocr, ocr_info = self.should_trigger_ocr(text, page)
        
        if should_ocr:
            pix = page.get_pixmap()
            ocr_metadata['ocr_reason'] = ocr_info['reason']

            try:
                if cv2 is not None and numpy is not None:
                    # 1. Convert PyMuPDF pixmap to numpy array
                    img_array = numpy.frombuffer(pix.samples, dtype=numpy.uint8).reshape(pix.height, pix.width, pix.n)

                    # 2. Convert to Grayscale
                    if pix.n >= 3:
                         gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                         gray = img_array

                    # 3. Apply Otsu's thresholding for Tesseract only
                    if self.ocr_engine in ["easyocr", "rapidocr"]:
                         processed_img = gray
                    else:
                         _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                else:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    processed_img = numpy.array(img) if 'numpy' in sys.modules else img


                if self.ocr_engine == "rapidocr":
                    try:
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
                        results = self.reader.readtext(
                            processed_img,
                            paragraph=True,
                            x_ths=0.8,
                            y_ths=0.5,
                            text_threshold=0.5,
                        )
                        text = ' '.join([res[1] for res in results])
                    except Exception as e:
                        print(f"EasyOCR error: {e}. Falling back to tesseract.")
                        pil_img = Image.fromarray(processed_img)
                        text = pytesseract.image_to_string(pil_img)
                else:
                    pil_img = Image.fromarray(processed_img) if not isinstance(processed_img, Image.Image) else processed_img
                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(pil_img, config=custom_config)
                
                ocr_metadata['ocr_used'] = True
                ocr_metadata['images_found'] = ocr_info.get('has_images', False)

            except Exception as e:
                print(f"OCR preprocessing error: {e}. Using native text.")
                ocr_metadata['ocr_error'] = str(e)

        return self.clean_text(text), ocr_metadata

    def identify_form_type(self, text: str) -> Tuple[str, bool, float, List[str], bool, List[str]]:
        """
        Identify the type of form based on text content using flexible regex.
        Returns a tuple of (form_type, is_start_page, confidence, matched_patterns, is_ambiguous, ambiguous_forms).
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

        # Track all detected forms for ambiguity detection
        detected_forms = []
        matched_patterns = []

        # Pass 0: Check for highly specific unique identifiers (Catalog numbers)
        # Use FUZZY matching to handle OCR errors
        text_lower = text.lower()
        forced_form_type = None
        best_catalog_match = None
        best_catalog_confidence = 0.0
        
        for form_type, patterns in self.unique_identifiers.items():
            for pattern in patterns:
                # Extract catalog number from regex pattern (remove \b and special regex chars)
                catalog = pattern.strip(r'\b')
                
                # Skip complex regex patterns (those with parentheses for grouping)
                if any(c in catalog for c in ['(', ')', '[', ']', '-']):
                    # Fall back to exact regex match for complex patterns
                    match = re.search(pattern, text_lower)
                    if match:
                        print(f"DEBUG: Found unique identifier (exact) '{pattern}' for {form_type}.")
                        forced_form_type = form_type
                        best_catalog_match = pattern
                        best_catalog_confidence = self.CONFIDENCE_UNIQUE_ID
                        break
                else:
                    # Use fuzzy matching for simple alphanumeric catalogs
                    found, confidence, matched_variant = self.fuzzy_match_catalog(text_lower, catalog)
                    if found:
                        print(f"DEBUG: Found catalog '{catalog}' for {form_type} with confidence {confidence} (matched: '{matched_variant}')")
                        forced_form_type = form_type
                        best_catalog_match = f"{catalog}→{matched_variant}"
                        best_catalog_confidence = confidence
                        # Higher confidence = higher priority, stop if we found a high-confidence match
                        if confidence >= 95:
                            break
            if forced_form_type and best_catalog_confidence >= 95:
                break

        # Helper to build return value
        def build_result(matched_type, is_start, confidence, patterns, is_ambiguous=False, ambiguous=None):
            # Use unique ID to override if found with good confidence
            if forced_form_type and matched_type not in ["OTHER", "CERTIFICATE"] and best_catalog_confidence >= 70:
                final_type = forced_form_type
                if final_type != matched_type:
                    print(f"DEBUG: Overriding matched type '{matched_type}' with forced type '{final_type}' based on catalog match (confidence: {best_catalog_confidence}).")
                confidence = max(confidence, best_catalog_confidence)
            else:
                final_type = matched_type
            return (final_type, is_start, confidence, patterns, is_ambiguous, ambiguous or [])

        # First pass: Check for titles (Strong match) using flexible regex
        earliest_title_type = None
        earliest_title_idx = float('inf')
        title_matches = []

        for form_type, _ in sorted_forms:
            pattern = title_patterns.get(form_type)
            if pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.start() < earliest_title_idx:
                    earliest_title_idx = match.start()
                    earliest_title_type = form_type
                    title_matches.append(form_type)
                    detected_forms.append((form_type, self.CONFIDENCE_TITLE, 'title'))

        if earliest_title_type:
            print(f"DEBUG: Found title pattern for {earliest_title_type} at idx {earliest_title_idx}")
            is_ambiguous = len(title_matches) > 1
            ambiguous = title_matches[1:] if is_ambiguous else []
            patterns_list = [f"title:{earliest_title_type}"]
            if best_catalog_match:
                patterns_list.insert(0, f"catalog:{best_catalog_match}")
            return build_result(earliest_title_type, True, self.CONFIDENCE_TITLE,
                              patterns_list, is_ambiguous, ambiguous)

        # Second pass: Check for 'Form X' fallbacks (Weak match) using flexible regex
        first_1000 = text[:1000].lower()
        earliest_match_type = None
        earliest_match_idx = float('inf')
        form_matches = []

        for form_type, _ in sorted_forms:
            clean_form = form_type.replace('-', '')
            regex_str = r'\s*[-]?\s*'.join(list(clean_form))

            match_form = re.search(rf'form\s*[-:\n]?\s*{regex_str}\b', first_1000, re.IGNORECASE)
            if match_form:
                if match_form.start() < earliest_match_idx:
                    earliest_match_idx = match_form.start()
                    earliest_match_type = form_type
                form_matches.append(form_type)
                detected_forms.append((form_type, self.CONFIDENCE_FORM, 'form_name'))

            match_standalone = re.search(rf'\b{regex_str}\b', first_1000, re.IGNORECASE)
            if match_standalone:
                if match_standalone.start() < earliest_match_idx:
                    earliest_match_idx = match_standalone.start()
                    earliest_match_type = form_type

        if earliest_match_type:
            is_start = earliest_match_idx < 300
            is_ambiguous = len(set(form_matches)) > 1
            ambiguous = list(set(form_matches) - {earliest_match_type}) if is_ambiguous else []
            print(f"DEBUG: Found '{earliest_match_type}' as the earliest form mention via flexible regex (idx: {earliest_match_idx}, is_start: {is_start})")
            if is_ambiguous:
                print(f"DEBUG: Ambiguous page - also found: {ambiguous}")
            patterns_list = [f"form_name:{earliest_match_type}"]
            if best_catalog_match:
                patterns_list.insert(0, f"catalog:{best_catalog_match}")
            return build_result(earliest_match_type, is_start, self.CONFIDENCE_FORM,
                              patterns_list, is_ambiguous, ambiguous)


        # Third pass: Fuzzy Matching (Safety net for very bad OCR)
        earliest_fuzzy_type = None
        earliest_fuzzy_idx = float('inf')
        is_fuzzy_title_match = False
        fuzzy_matches = []

        for form_type, title in sorted_forms:
            form_string = f"form {form_type.lower()}"
            score_form = fuzz.partial_ratio(form_string, first_1000)

            if score_form > 90:
                clean_form = form_type.replace('-', '')
                regex_str = r'\s*[-]?\s*'.join(list(clean_form))
                match = re.search(rf'form\s*[-:\n]?\s*{regex_str}\b', first_1000, re.IGNORECASE)
                idx = match.start() if match else float('inf')

                if idx < earliest_fuzzy_idx:
                    earliest_fuzzy_idx = idx
                    earliest_fuzzy_type = form_type
                    is_fuzzy_title_match = False
                fuzzy_matches.append(form_type)
                detected_forms.append((form_type, self.CONFIDENCE_FUZZY, 'fuzzy'))

            score_title = fuzz.partial_ratio(title.lower(), first_1000)
            if score_title > 85:
                print(f"DEBUG: Found title via fuzzy matching (Score: {score_title}) for {form_type}")
                patterns_list = [f"fuzzy_title:{form_type}"]
                if best_catalog_match:
                    patterns_list.insert(0, f"catalog:{best_catalog_match}")
                return build_result(form_type, True, self.CONFIDENCE_FUZZY, patterns_list)

        if earliest_fuzzy_type:
            is_start = is_fuzzy_title_match or (earliest_fuzzy_idx < 300)
            is_ambiguous = len(set(fuzzy_matches)) > 1
            ambiguous = list(set(fuzzy_matches) - {earliest_fuzzy_type}) if is_ambiguous else []
            print(f"DEBUG: Found '{earliest_fuzzy_type}' as the earliest form mention via fuzzy matching (idx: {earliest_fuzzy_idx}, is_start: {is_start})")
            patterns_list = [f"fuzzy:{earliest_fuzzy_type}"]
            if best_catalog_match:
                patterns_list.insert(0, f"catalog:{best_catalog_match}")
            return build_result(earliest_fuzzy_type, is_start, self.CONFIDENCE_FUZZY,
                              patterns_list, is_ambiguous, ambiguous)

        # Check for certificates
        if self.certificate_pattern.search(text):
            patterns_list = ["certificate_pattern"]
            if best_catalog_match:
                patterns_list.insert(0, f"catalog:{best_catalog_match}")
            return build_result("CERTIFICATE", True, self.CONFIDENCE_CERTIFICATE, patterns_list)

        return build_result("OTHER", False, 50.0, [], False, [])

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

    def analyze_page(self, page_num: int, page) -> PageAnalysis:
        """Analyze a single page and return structured analysis."""
        text, ocr_metadata = self.extract_text_from_page(page)
        
        # Try LLM classification first if enabled
        llm_result = None
        if self.use_llm and self.client:
            llm_result = self.classify_page_with_llm(text, page_num)
        
        if llm_result:
            # Use LLM result
            return PageAnalysis(
                page_num=page_num,
                text=text,
                form_type=llm_result['form_type'],
                is_start_page=llm_result['is_start_page'],
                confidence=llm_result['confidence'],
                has_images=ocr_metadata.get('images_found', False),
                text_length=len(text),
                matched_patterns=llm_result['matched_patterns'],
                is_ambiguous=llm_result['is_ambiguous'],
                ambiguous_forms=llm_result.get('ambiguous_forms', [])
            )
        else:
            # Fall back to logic-based classification
            form_type, is_start, confidence, patterns, is_ambiguous, ambiguous_forms = self.identify_form_type(text)
            return PageAnalysis(
                page_num=page_num,
                text=text,
                form_type=form_type,
                is_start_page=is_start,
                confidence=confidence,
                has_images=ocr_metadata.get('images_found', False),
                text_length=len(text),
                matched_patterns=patterns,
                is_ambiguous=is_ambiguous,
                ambiguous_forms=ambiguous_forms or []
            )

    def analyze_pages_parallel(self, pdf_document, max_workers: int = 4) -> List[PageAnalysis]:
        """Analyze all pages in parallel using ThreadPoolExecutor."""
        analyses = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page analysis tasks
            future_to_page = {
                executor.submit(self.analyze_page, page_num, pdf_document[page_num]): page_num
                for page_num in range(len(pdf_document))
            }
            
            # Collect results as they complete (may not be in order)
            results = {}
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    analysis = future.result()
                    results[page_num] = analysis
                except Exception as e:
                    print(f"Error analyzing page {page_num}: {e}")
                    # Create a fallback analysis
                    results[page_num] = PageAnalysis(
                        page_num=page_num,
                        text="",
                        form_type="OTHER",
                        is_start_page=False,
                        confidence=0.0,
                        has_images=False,
                        text_length=0,
                        matched_patterns=[],
                        is_ambiguous=False,
                        ambiguous_forms=None
                    )
            
            # Convert to ordered list
            analyses = [results[i] for i in range(len(results))]
        
        return analyses

    def group_pages(self, pdf_document, use_parallel: bool = True, max_workers: int = 4) -> List[Dict]:
        """Group pages into separate documents strictly by physical page count."""
        # Analyze all pages (parallel or sequential)
        if use_parallel:
            print(f"Analyzing {len(pdf_document)} pages in parallel with {max_workers} workers...")
            analyses = self.analyze_pages_parallel(pdf_document, max_workers)
        else:
            print(f"Analyzing {len(pdf_document)} pages sequentially...")
            analyses = []
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                analysis = self.analyze_page(page_num, page)
                analyses.append(analysis)
        
        # Debug print analysis results
        for analysis in analyses:
            print(f"DEBUG: Page {analysis.page_num} -> Type: {analysis.form_type}, "
                  f"IsStart: {analysis.is_start_page}, Confidence: {analysis.confidence}, "
                  f"Ambiguous: {analysis.is_ambiguous}")
            if analysis.is_ambiguous:
                print(f"  WARNING: Page {analysis.page_num} is ambiguous - also contains: {analysis.ambiguous_forms}")

        documents = []
        current_doc = None

        for analysis in analyses:
            form_type = analysis.form_type
            is_start_page = analysis.is_start_page
            page_num = analysis.page_num
            
            start_new = False

            if current_doc is None:
                start_new = True
            else:
                current_type = current_doc['type']
                current_len = len(current_doc['pages'])

                # Check if we are currently building a form with known strict length rules
                if current_type in self.form_rules:
                    allowed_counts = self.form_rules[current_type]
                    max_count = max(allowed_counts)

                    if current_len < max_count:
                        # We have NOT reached the maximum allowed length for this form yet.
                        if current_len in allowed_counts and is_start_page and form_type != "OTHER" and form_type != current_type:
                            start_new = True
                        else:
                            # Blindly absorb page
                            start_new = False
                    else:
                        # Hit max length, force new doc
                        start_new = True

                else:
                    # No strict length rules
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
                    'text': analysis.text,
                    'confidence': analysis.confidence,
                    'is_ambiguous': analysis.is_ambiguous,
                    'ambiguous_forms': analysis.ambiguous_forms or []
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

    def split_pdf(self, input_path: str, output_dir: str, use_parallel: bool = True, max_workers: int = 4) -> List[Dict]:
        """
        Split PDF into separate files based on form types.
        
        Args:
            input_path: Path to input PDF
            output_dir: Directory to save split PDFs
            use_parallel: Whether to use parallel processing (default True)
            max_workers: Number of worker threads for parallel processing (default 4)
        
        Returns:
            List of document dictionaries with metadata
        """
        os.makedirs(output_dir, exist_ok=True)

        # Open the PDF
        pdf_document = fitz.open(input_path)

        # Group pages into separate documents
        documents = self.group_pages(pdf_document, use_parallel=use_parallel, max_workers=max_workers)

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

    def get_low_confidence_documents(self, documents: List[Dict], threshold: float = 80.0) -> List[Dict]:
        """Return documents with confidence below threshold."""
        return [doc for doc in documents if doc.get('confidence', 100.0) < threshold]

    def get_ambiguous_documents(self, documents: List[Dict]) -> List[Dict]:
        """Return documents that contain ambiguous pages."""
        return [doc for doc in documents if doc.get('is_ambiguous', False)]

def main():
    # You can change to "easyocr" to test locally
    splitter = PDFSplitter(ocr_engine="tesseract")
    input_pdf = "ooo.pdf"
    output_directory = "split_forms"

    try:
        result = splitter.split_pdf(input_pdf, output_directory, use_parallel=True, max_workers=4)

        # Print summary
        print("\nPDF splitting complete! Summary:")
        for doc in result:
            confidence = doc.get('confidence', 'N/A')
            ambiguous = "⚠️ Ambiguous" if doc.get('is_ambiguous', False) else ""
            print(f"{doc['type']} (ID: {doc['id']}): {len(doc['pages'])} page(s) | Confidence: {confidence}% {ambiguous}")

        # Show warnings
        low_conf = splitter.get_low_confidence_documents(result)
        ambiguous_docs = splitter.get_ambiguous_documents(result)
        
        if low_conf:
            print(f"\n⚠️ Low confidence documents: {[d['type'] + '#' + str(d['id']) for d in low_conf]}")
        if ambiguous_docs:
            print(f"\n⚠️ Ambiguous documents: {[d['type'] + '#' + str(d['id']) for d in ambiguous_docs]}")

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()