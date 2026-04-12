import os
import sys
import json
import time
import random
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
import pytesseract
from PIL import Image
try:
    import cv2
    import numpy
    cv2.setNumThreads(4) # Force OpenCV to use multiple CPU cores for preprocessing
    cv2.setUseOptimized(True) # Force SIMD/Intel optimizations
except ImportError:
    cv2 = None
    numpy = None
import re
from collections import defaultdict
from thefuzz import fuzz

# Import LLM configuration
from llm_config import LLMConfig, default_config, OCR_SYSTEM_PROMPT, CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_TEMPLATE

# Optional LLM support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# LangChain for LLM-based OCR
try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain_core.messages import HumanMessage
    import base64
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    AzureChatOpenAI = None
    ChatOpenAI = None
    HumanMessage = None
    base64 = None


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
    reasoning: str = ''
    page_info: str = None
    form_structure_detected: bool = False

class PDFSplitter:
    """
    PDF Splitter with LLM-powered classification.
    
    CLASSIFICATION MODES:
    
    1. LLM Mode (Primary, when use_llm=True):
       - Sends FULL page text to GPT-4o mini (or custom LLM)
       - LLM analyzes complete context holistically
       - LLM decides document type based on structure, content, context
       - NO hardcoded rules or priorities
       - Confidence scores come from LLM (0-100%)
       
    2. Logic-Based Mode (Fallback, when LLM disabled):
       - Uses regex patterns and fuzzy matching
       - Confidence thresholds (for internal tracking only):
         * Catalog match: 70-100% (fuzzy)
         * Title match: 95%
         * Form name: 80%
         * Fuzzy match: 60%
         * Certificate: 70%
       - These are NOT used to influence LLM decisions
    """
    
    # Confidence thresholds for logic-based detection (fallback only)
    # These are NOT used when LLM classification is enabled
    CONFIDENCE_CERTIFICATE = 70.0  # Certificate pattern (fallback)
    
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

    # Note: LLM prompts moved to llm_config.py for easy customization

    def __init__(self, ocr_engine="tesseract", use_llm=False, api_key=None, llm_model="gpt-4o-mini",
                 use_llm_ocr=False, azure_deployment=None, azure_endpoint=None, azure_api_version=None,
                 llm_config: Optional[LLMConfig] = None):
        """
        Initialize PDFSplitter.

        Args:
            ocr_engine: OCR engine to use ("tesseract", "easyocr", "rapidocr", or "llm")
            use_llm: Enable LLM-based classification (uses text input)
            use_llm_ocr: Enable LLM-based OCR using vision models (requires LangChain)
            api_key: Azure OpenAI API key
            llm_model: Model name for classification (e.g., "gpt-4o-mini")
            azure_deployment: Azure deployment name
            azure_endpoint: Azure endpoint URL (e.g., "https://your-resource.openai.azure.com/")
            azure_api_version: Azure API version (e.g., "2024-02-15-preview")
            llm_config: Optional LLMConfig instance for advanced configuration.
                       If provided, all parameters will be overridden by this config.
        """
        self.ocr_engine = ocr_engine
        
        # Use provided config or create one from parameters
        if llm_config:
            self.llm_config = llm_config
        else:
            # Create config from parameters for backward compatibility
            self.llm_config = LLMConfig(
                api_key=api_key,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version or "2024-02-15-preview",
                ocr_model="gpt-4o",
                classification_model=llm_model,
                ocr_image_zoom=2.0
            )
        
        self.use_llm_ocr = use_llm_ocr
        self.use_llm = use_llm

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
            "W-9": "Request for Taxpayer Identification Number",
            "WITHHOLDING STATEMENT": "Withholding Statement"
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

        # LLM Configuration from llm_config
        self.api_key = self.llm_config.api_key
        self.api_base = None  # Not used with Azure
        self.llm_model = self.llm_config.classification_model
        self.client = None
        self.llm_ocr_client = None

        if self.use_llm:
            if not OPENAI_AVAILABLE:
                print("Warning: openai package not installed. LLM classification disabled.")
                self.use_llm = False
            elif not self.api_key:
                print("Warning: No API key provided. LLM classification disabled.")
                self.use_llm = False
            else:
                try:
                    # Use OpenAI client with Azure configuration
                    # Azure OpenAI requires specific base URL format
                    azure_base_url = f"{self.llm_config.azure_endpoint.rstrip('/')}/openai/deployments/{self.llm_config.azure_deployment}"
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url=azure_base_url,
                        default_headers={"api-key": self.api_key}
                    )
                    print(f"LLM classification enabled using Azure deployment: {self.llm_config.azure_deployment}")
                except Exception as e:
                    print(f"Warning: Failed to initialize OpenAI client: {e}. LLM classification disabled.")
                    self.use_llm = False

        # Initialize LangChain client for LLM-based OCR (Azure only)
        if self.use_llm_ocr:
            if not LANGCHAIN_AVAILABLE:
                print("Warning: langchain-openai not installed. LLM OCR disabled.")
                self.use_llm_ocr = False
            elif not self.api_key:
                print("Warning: No API key provided. LLM OCR disabled.")
                self.use_llm_ocr = False
            else:
                try:
                    # Azure OpenAI
                    self.llm_ocr_client = AzureChatOpenAI(
                        azure_deployment=self.llm_config.azure_deployment,
                        azure_endpoint=self.llm_config.azure_endpoint,
                        api_key=self.api_key,
                        api_version=self.llm_config.azure_api_version,
                        model=self.llm_config.ocr_model,
                        max_tokens=None,
                        temperature=0.0
                    )
                    print(f"LLM OCR enabled using Azure deployment: {self.llm_config.azure_deployment}")
                except Exception as e:
                    print(f"Warning: Failed to initialize LangChain client: {e}. LLM OCR disabled.")
                    self.use_llm_ocr = False
    
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
        Use LLM to classify a page based on its COMPLETE text content.
        The LLM will analyze the full context and decide what document type this page belongs to.

        Args:
            text: Extracted text from the page (full text, no truncation)
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with classification results, or None if LLM fails
        """
        if not self.use_llm or not self.client:
            return None

        try:
            # Send FULL page text - no truncation
            # LLM should see complete context to make informed decision
            # Note: We still have a safety limit for extremely large pages
            max_safety_limit = 50000  # 50K chars = ~12.5K tokens, well within GPT-4o-mini's 128K limit
            if len(text) > max_safety_limit:
                print(f"Warning: Page {page_num} has {len(text)} chars, truncating to {max_safety_limit}")
                text = text[:max_safety_limit] + "\n...[truncated due to extreme length]..."

            # Build the prompt
            user_prompt = CLASSIFICATION_USER_TEMPLATE.format(
                page_num=page_num + 1,  # 1-indexed for LLM
                text=text
            )

            # Call the LLM with retry logic
            def _do_classify():
                return self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": CLASSIFICATION_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.0,  # Zero temperature for deterministic consistency
                    timeout=60  # Longer timeout for large pages
                )

            response = self._llm_call_with_retry(
                _do_classify,
                label=f"classification page {page_num + 1}",
                max_retries=3,
                base_delay=2.0,
                max_delay=30.0
            )

            # Parse the response
            response_text = response.choices[0].message.content.strip()

            # Try to extract JSON from the response
            # Handle cases where LLM adds markdown code blocks or extra text
            if response_text.startswith("```"):
                # Remove markdown code block if present
                lines = response_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines)

            # Find JSON in response (in case there's extra text before/after)
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]

            result = json.loads(response_text)
            
            # Map LLM response to our format
            document_type = result.get("document_type", "OTHER").upper()
            confidence = float(result.get("confidence", 0.5)) * 100
            reasoning = result.get("reasoning", "LLM classification")
            is_first_page = result.get("is_first_page", False)
            form_structure_detected = result.get("form_structure_detected", False)
            mentions_other_forms = result.get("mentions_other_forms", [])
            
            # Ensure mentions_other_forms is strictly a list (Fix for JSON drift)
            if isinstance(mentions_other_forms, str):
                mentions_other_forms = [mentions_other_forms]
            elif not isinstance(mentions_other_forms, list):
                mentions_other_forms = []
            
            # Validate document type
            valid_types = ["W-8BEN", "W-8BEN-E", "W-8EXP", "W-8IMY", "W-9",
                           "CERTIFICATE", "WITHHOLDING STATEMENT", "DOCUSIGN", "OTHER"]
            if document_type not in valid_types:
                print(f"Warning: LLM returned invalid document_type '{document_type}', defaulting to OTHER")
                document_type = "OTHER"
            
            # Build patterns list
            patterns = [f"llm:{document_type}"]
            if form_structure_detected:
                patterns.append("llm_form_structure:true")
            if mentions_other_forms:
                patterns.append(f"llm_mentions:{','.join(mentions_other_forms)}")
            
            # Check for ambiguity
            is_ambiguous = len(mentions_other_forms) > 0 and document_type != "OTHER"
            
            return {
                'form_type': document_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'is_start_page': is_first_page,
                'matched_patterns': patterns,
                'is_ambiguous': is_ambiguous,
                'ambiguous_forms': mentions_other_forms if is_ambiguous else [],
                'form_structure_detected': form_structure_detected,
                'method': 'llm'
            }
            
        except json.JSONDecodeError as e:
            # Model returned malformed JSON — not a transient error, skip retry
            print(f"[Classification] JSON parse error on page {page_num + 1}: {e}. "
                  f"Raw response: {response_text[:200]}...")
            return None
        except Exception as e:
            # All retries exhausted (retry helper already logged each attempt)
            print(f"[Classification] Failed permanently on page {page_num + 1}: {type(e).__name__}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        # Remove extra whitespace and newlines
        return ' '.join(text.replace('\n', ' ').split())

    def _llm_call_with_retry(self, call_fn: Callable[[], Any], label: str,
                              max_retries: int = 3,
                              base_delay: float = 2.0,
                              max_delay: float = 30.0) -> Any:
        """
        Execute an LLM API call with robust exponential backoff and jitter.

        Retries on transient errors:
          - HTTP 429 (rate limit / quota)
          - HTTP 5xx (server errors)
          - Timeout / connection errors

        Does NOT retry on:
          - HTTP 400 / 401 / 403 (bad request / auth — retrying won't help)
          - json.JSONDecodeError (model returned garbage — let caller handle)

        Args:
            call_fn:     Zero-argument callable that executes the actual API call.
            label:       Human-readable label for log messages (e.g. "classification page 3").
            max_retries: Maximum number of retry attempts after the first failure.
            base_delay:  Base delay in seconds before the first retry.
            max_delay:   Maximum allowed delay between retries.

        Returns:
            The return value of call_fn on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        # Error status codes that are worth retrying
        RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}
        # Error status codes that are definitely NOT worth retrying
        NON_RETRIABLE_STATUS_CODES = {400, 401, 403, 404}

        last_exception = None

        for attempt in range(max_retries + 1):  # attempt 0 = first try
            try:
                return call_fn()

            except json.JSONDecodeError:
                # Model returned malformed JSON — no point retrying the same call
                raise

            except Exception as e:
                last_exception = e
                error_name = type(e).__name__

                # Try to extract HTTP status code from the exception
                status_code = None
                if hasattr(e, 'status_code'):
                    status_code = e.status_code
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code

                # Check for non-retriable status codes immediately
                if status_code in NON_RETRIABLE_STATUS_CODES:
                    print(f"[LLM Retry] {label} — Non-retriable error {status_code} ({error_name}). Aborting.")
                    raise

                # If we've used all retries, give up
                if attempt >= max_retries:
                    print(f"[LLM Retry] {label} — All {max_retries} retries exhausted. Last error: {error_name}: {e}")
                    raise

                # Determine if this is a known retriable error
                is_retriable = (
                    status_code in RETRIABLE_STATUS_CODES
                    or status_code is None  # Unknown errors (timeout, connection reset) → retry
                )

                if not is_retriable:
                    print(f"[LLM Retry] {label} — Non-retriable error {status_code} ({error_name}). Aborting.")
                    raise

                # Compute exponential backoff with full jitter
                # Formula: random(0, min(max_delay, base_delay * 2^attempt))
                cap = min(max_delay, base_delay * (2 ** attempt))
                delay = random.uniform(0.5, cap)

                # If the API returned a Retry-After header, honour it
                retry_after = getattr(e, 'retry_after', None) or (
                    getattr(getattr(e, 'response', None), 'headers', {}) or {}
                ).get('Retry-After')
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except (ValueError, TypeError):
                        pass

                print(f"[LLM Retry] {label} — Attempt {attempt + 1}/{max_retries} failed "
                      f"({error_name}: {e}). Retrying in {delay:.1f}s...")
                time.sleep(delay)

        raise last_exception  # Should never reach here but satisfies type checkers
    
    def extract_text_with_llm_ocr(self, page, page_num: int) -> Tuple[str, dict]:
        """
        Extract text from a page using LLM vision model (GPT-4o or similar).
        Uses LangChain for OpenAI or Azure OpenAI.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        metadata = {
            'ocr_used': True,
            'ocr_method': 'llm_vision',
            'ocr_engine': 'langchain_openai'
        }

        if not self.use_llm_ocr or not self.llm_ocr_client:
            return "", metadata

        try:
            # Convert page to image using zoom level from config
            pix = page.get_pixmap(matrix=fitz.Matrix(
                self.llm_config.ocr_image_zoom,
                self.llm_config.ocr_image_zoom
            ))
            img_bytes = pix.tobytes("png")

            # Encode image to base64
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            # Call the LLM Vision with retry logic
            if HumanMessage:
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": OCR_SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }}
                    ]
                )

                def _do_ocr_call():
                    return self.llm_ocr_client.invoke([message])

                response = self._llm_call_with_retry(
                    _do_ocr_call,
                    label=f"vision OCR page {page_num + 1}",
                    max_retries=3,
                    base_delay=2.0,
                    max_delay=30.0
                )
                extracted_text = response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback if HumanMessage not available
                extracted_text = ""

            metadata['ocr_success'] = True
            metadata['text_length'] = len(extracted_text)

            return self.clean_text(extracted_text), metadata

        except Exception as e:
            # All retries exhausted (retry helper already logged each attempt)
            print(f"[Vision OCR] Failed permanently on page {page_num + 1}: {type(e).__name__}: {e}")
            metadata['ocr_error'] = str(e)
            metadata['ocr_success'] = False
            return "", metadata

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
        
        # We are intentionally forcing OCR on EVERY page to bypass phantom/corrupted native text layers.
        # This guarantees that the LLM Vision model (or other selected OCR engine) always processes the page.
        metadata['reason'] = 'forced_ocr_every_page'
        return True, metadata

    def extract_text_from_page(self, page) -> Tuple[str, dict]:
        """
        Extract text from a page using native text extraction, LLM OCR, or traditional OCR if needed.
        
        Priority:
        1. Native text extraction (if sufficient)
        2. LLM OCR (if use_llm_ocr is enabled)
        3. Traditional OCR (Tesseract/EasyOCR/RapidOCR) as fallback
        
        Returns (text, metadata_dict).
        """
        # Try native text extraction first
        text = page.get_text()
        ocr_metadata = {'ocr_used': False, 'ocr_reason': None}

        # Determine if OCR should be triggered
        should_ocr, ocr_info = self.should_trigger_ocr(text, page)

        if should_ocr:
            # If LLM OCR is enabled, use it first
            if self.use_llm_ocr and self.llm_ocr_client:
                llm_ocr_text, llm_metadata = self.extract_text_with_llm_ocr(page, page.number)
                if llm_ocr_text:
                    ocr_metadata.update(llm_metadata)
                    ocr_metadata['ocr_reason'] = ocr_info['reason']
                    return llm_ocr_text, ocr_metadata
                else:
                    print(f"Page {page.number}: LLM OCR failed, falling back to traditional OCR")
                    ocr_metadata['llm_ocr_attempted'] = True
            
            # Fall back to traditional OCR
            pix = page.get_pixmap()
            ocr_metadata['ocr_reason'] = ocr_info['reason']

            try:
                if cv2 is not None and numpy is not None:
                    # 1. Convert PyMuPDF pixmap to PNG bytes directly, then securely decode into grayscale matrix
                    img_array = numpy.frombuffer(pix.tobytes("png"), dtype=numpy.uint8)
                    gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                    # 2. Apply Otsu's thresholding for Tesseract only
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
            "W-9": r'request.*?taxpayer\s*identification\s*number',
            "WITHHOLDING STATEMENT": r'withholding\s*statement'
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
                        best_catalog_confidence = 100.0  # Exact match = high confidence
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
            if forced_form_type and matched_type not in ["OTHER", "CERTIFICATE", "WITHHOLDING STATEMENT"] and best_catalog_confidence >= 70:
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
                    detected_forms.append((form_type, 95.0, 'title'))  # Title match = 95% confidence

        if earliest_title_type:
            print(f"DEBUG: Found title pattern for {earliest_title_type} at idx {earliest_title_idx}")
            is_ambiguous = len(title_matches) > 1
            ambiguous = title_matches[1:] if is_ambiguous else []
            patterns_list = [f"title:{earliest_title_type}"]
            if best_catalog_match:
                patterns_list.insert(0, f"catalog:{best_catalog_match}")
            return build_result(earliest_title_type, True, 95.0,  # Title match confidence
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
                detected_forms.append((form_type, 80.0, 'form_name'))  # Form name match = 80% confidence

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
            return build_result(earliest_match_type, is_start, 80.0,  # Form name confidence
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
                detected_forms.append((form_type, 60.0, 'fuzzy'))  # Fuzzy match = 60% confidence

            score_title = fuzz.partial_ratio(title.lower(), first_1000)
            if score_title > 85:
                print(f"DEBUG: Found title via fuzzy matching (Score: {score_title}) for {form_type}")
                patterns_list = [f"fuzzy_title:{form_type}"]
                if best_catalog_match:
                    patterns_list.insert(0, f"catalog:{best_catalog_match}")
                return build_result(form_type, True, 60.0, patterns_list)  # Fuzzy title confidence

        if earliest_fuzzy_type:
            is_start = is_fuzzy_title_match or (earliest_fuzzy_idx < 300)
            is_ambiguous = len(set(fuzzy_matches)) > 1
            ambiguous = list(set(fuzzy_matches) - {earliest_fuzzy_type}) if is_ambiguous else []
            print(f"DEBUG: Found '{earliest_fuzzy_type}' as the earliest form mention via fuzzy matching (idx: {earliest_fuzzy_idx}, is_start: {is_start})")
            patterns_list = [f"fuzzy:{earliest_fuzzy_type}"]
            if best_catalog_match:
                patterns_list.insert(0, f"catalog:{best_catalog_match}")
            return build_result(earliest_fuzzy_type, is_start, 60.0,  # Fuzzy match confidence
                              patterns_list, is_ambiguous, ambiguous)

        # Check for certificates
        if self.certificate_pattern.search(text):
            patterns_list = ["certificate_pattern"]
            if best_catalog_match:
                patterns_list.insert(0, f"catalog:{best_catalog_match}")
            return build_result("CERTIFICATE", True, 70.0, patterns_list)  # Certificate confidence

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
                print(f"✓ Page {page_num}: LLM classified as {llm_result['form_type']} (confidence: {llm_result['confidence']:.0f}%)")
                print(f"  Reasoning: {llm_result['reasoning'][:150]}...")
            else:
                print(f"⚠ Page {page_num}: LLM failed, falling back to logic-based classification")

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
                ambiguous_forms=llm_result.get('ambiguous_forms', []),
                reasoning=llm_result.get('reasoning', ''),
                page_info=None,
                form_structure_detected=llm_result.get('form_structure_detected', False)
            )
        else:
            # Fall back to logic-based classification
            form_type, is_start, confidence, patterns, is_ambiguous, ambiguous_forms = self.identify_form_type(text)
            print(f"→ Page {page_num}: Logic classified as {form_type} (confidence: {confidence:.0f}%)")
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
                ambiguous_forms=ambiguous_forms or [],
                reasoning='',
                page_info=None,
                form_structure_detected=False
            )

    def _analyze_page_from_path(self, path: str, page_num: int) -> PageAnalysis:
        """Open document locally in thread and analyze a single page (PyMuPDF thread-safety fix)."""
        doc = fitz.open(path)
        try:
            page = doc[page_num]
            return self.analyze_page(page_num, page)
        finally:
            doc.close()

    def analyze_pages_parallel(self, pdf_document, input_path: str = None, max_workers: int = 4) -> List[PageAnalysis]:
        """Analyze all pages in parallel using ThreadPoolExecutor."""
        analyses = []
        
        # PyMuPDF requires each thread to have its own Document instance
        path_to_open = input_path or pdf_document.name
        total_pages = len(pdf_document)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page analysis tasks
            future_to_page = {
                executor.submit(self._analyze_page_from_path, path_to_open, page_num): page_num
                for page_num in range(total_pages)
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
                        ambiguous_forms=None,
                        reasoning='',
                        page_info=None,
                        form_structure_detected=False
                    )
            
            # Convert to ordered list
            analyses = [results[i] for i in range(len(results))]
        
        return analyses

    def group_pages(self, pdf_document, input_path: str = None, use_parallel: bool = True, max_workers: int = 4) -> List[Dict]:
        """Group pages into separate documents strictly by physical page count."""
        # Analyze all pages (parallel or sequential)
        if use_parallel:
            print(f"Analyzing {len(pdf_document)} pages in parallel with {max_workers} workers...")
            analyses = self.analyze_pages_parallel(pdf_document, input_path=input_path, max_workers=max_workers)
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
        # Page budget: tracks how many more pages are expected for the current document.
        # -1 means "unknown / no budget set" (e.g. for OTHER / WITHHOLDING STATEMENT)
        current_budget = -1
        # Flag: a DocuSign certificate was just absorbed into the current form.
        # The very next page after a DocuSign MUST start a new document.
        docusign_just_absorbed = False
        # Form types that can have a DocuSign certificate appended to them
        DOCUSIGN_ELIGIBLE_TYPES = set(self.form_rules.keys()) | {"WITHHOLDING STATEMENT"}

        for analysis in analyses:
            form_type = analysis.form_type
            is_start_page = analysis.is_start_page
            page_num = analysis.page_num

            start_new = False

            # -------------------------------------------------------
            # DOCUSIGN GATE: if a DocuSign was just absorbed, the page
            # after it MUST begin a fresh document unconditionally.
            # -------------------------------------------------------
            if docusign_just_absorbed:
                start_new = True
                docusign_just_absorbed = False
                print(f"DEBUG: Page {page_num} starts new doc (follows DocuSign certificate)")

            elif current_doc is None:
                start_new = True
            else:
                current_type = current_doc['type']
                current_len = len(current_doc['pages'])

                # -------------------------------------------------------
                # DOCUSIGN ABSORPTION: A DocuSign Certificate of Completion
                # always physically follows the form it belongs to.
                # When we encounter a DOCUSIGN page, absorb it into the
                # current document (regardless of budget state) and set
                # the flag so the page after it starts a new document.
                # DocuSign pages do NOT count against the form's page budget.
                # -------------------------------------------------------
                if form_type == "DOCUSIGN":
                    if current_type in DOCUSIGN_ELIGIBLE_TYPES:
                        start_new = False
                        docusign_just_absorbed = True
                        print(f"DEBUG: Page {page_num} is DocuSign certificate — absorbing into {current_type} document")
                    else:
                        # DocuSign after OTHER/CERTIFICATE — treat as standalone OTHER
                        start_new = True
                        print(f"DEBUG: Page {page_num} is DocuSign but current doc is {current_type} — starting new")

                # -------------------------------------------------------
                # KEY RULE: If we have an active page budget (i.e. we are
                # mid-way through a known-length form), absorb this page
                # unconditionally — regardless of what the LLM said about
                # its type.  The LLM often labels interior instruction pages
                # as OTHER even though they are physically part of the form.
                # The pre-verified guarantee means these pages MUST belong
                # to the current document until the budget is exhausted.
                # -------------------------------------------------------
                elif current_budget > 0:
                    start_new = False
                    print(f"DEBUG: Page {page_num} absorbed into {current_type} (budget remaining: {current_budget})")

                # Different form type and no active budget → start new doc
                elif form_type != current_type:
                    start_new = True

                # Same type / same category checks
                elif form_type == "OTHER" or current_type == "OTHER":
                    start_new = False
                elif form_type == "CERTIFICATE" or current_type == "CERTIFICATE":
                    start_new = True
                elif current_type in self.form_rules:
                    allowed_counts = self.form_rules[current_type]
                    max_count = max(allowed_counts)

                    if current_len >= max_count:
                        start_new = True
                        print(f"DEBUG: Force split - {current_type} reached max page count ({current_len} >= {max_count})")
                    elif current_len in allowed_counts and is_start_page:
                        start_new = True
                        print(f"DEBUG: Split at valid count ({current_len}) with is_start_page=True")
                    else:
                        start_new = False
                else:
                    start_new = False

            if start_new:
                if current_doc is not None:
                    documents.append(current_doc)
                    print(f"DEBUG: Completed document: {current_doc['type']} #{len([d for d in documents if d['type'] == current_doc['type']])} with {len(current_doc['pages'])} page(s)")

                # Create new document
                doc_id = len([d for d in documents if d['type'] == form_type]) + 1
                current_doc = {
                    'type': form_type,
                    'pages': [],
                    'id': doc_id,
                    'text': analysis.text,
                    'confidence': analysis.confidence,
                    'is_ambiguous': analysis.is_ambiguous,
                    'ambiguous_forms': analysis.ambiguous_forms or [],
                    'matched_patterns': analysis.matched_patterns,
                    'method': 'llm' if any('llm' in p for p in analysis.matched_patterns) else 'logic',
                    'reasoning': analysis.reasoning,
                    'form_structure_detected': analysis.form_structure_detected
                }

                # Set the page budget for this new document.
                # For forms with known page counts, we lock in the budget based on
                # is_first_page. If the LLM says this is a first page, we don't yet
                # know whether it's the 1-page or 6-page variant of W-9, so we start
                # with -1 (dynamic). The budget will be locked in on page 2 if a
                # continuation page is absorbed (meaning it must be the max variant).
                # For forms with only ONE allowed count, set it immediately.
                if form_type in self.form_rules:
                    allowed_counts = self.form_rules[form_type]
                    if len(allowed_counts) == 1:
                        # Single allowed length (e.g. W-8BEN=1, W-8IMY=8) — lock immediately
                        current_budget = allowed_counts[0]
                    else:
                        # Multiple allowed lengths (e.g. W-9=[1,6]) — start undecided
                        current_budget = -1
                else:
                    current_budget = -1  # No rule → no budget

                print(f"DEBUG: Starting new document: {form_type} #{doc_id} (initial budget: {current_budget})")

            current_doc['pages'].append(page_num)

            # Decrement the budget now that we've consumed one page.
            # DocuSign pages are NOT counted against the form budget.
            if form_type == "DOCUSIGN":
                pass  # DocuSign never affects the budget
            elif current_budget > 0:
                current_budget -= 1
                print(f"DEBUG: Page {page_num} consumed. Budget now: {current_budget}")
            elif current_budget == -1 and form_type in self.form_rules:
                # Budget is undecided. Check if we just absorbed a CONTINUATION page
                # (is_first_page=False) — this locks us into the max allowed count.
                if not is_start_page and len(current_doc['pages']) == 2:
                    max_count = max(self.form_rules[form_type])
                    remaining = max_count - len(current_doc['pages'])
                    current_budget = remaining
                    print(f"DEBUG: Continuation page detected for {form_type}. Locking budget to {max_count} pages. Remaining: {current_budget}")
                elif is_start_page and len(current_doc['pages']) == 1:
                    # First page seen, could be 1-page variant — do nothing yet
                    pass

            # Update dynamic confidence - Always take the minimum confidence across all pages
            if analysis.confidence < current_doc['confidence']:
                current_doc['confidence'] = analysis.confidence

            # Update dynamic ambiguity - Any ambiguous page makes the document ambiguous
            if analysis.is_ambiguous:
                current_doc['is_ambiguous'] = True
                if analysis.ambiguous_forms:
                    current_doc['ambiguous_forms'].extend([f for f in analysis.ambiguous_forms if f not in current_doc['ambiguous_forms']])

        # Add the last document
        if current_doc is not None:
            documents.append(current_doc)
            print(f"DEBUG: Completed final document: {current_doc['type']} with {len(current_doc['pages'])} page(s)")

        return documents

    def generate_filename(self, doc: Dict) -> str:
        """Generate appropriate filename for the document."""
        if doc['type'] == "CERTIFICATE":
            # Extract a meaningful name from the certificate text
            text = doc['text'].lower()
            cert_idx = text.find("certificate")
            if cert_idx != -1:
                words = text.split()
                cert_word_idx = next(i for i, word in enumerate(words) if "certificate" in word.lower())
                start_idx = max(0, cert_word_idx - 5)
                end_idx = min(len(words), cert_word_idx + 6)
                cert_name = "_".join(words[start_idx:end_idx])
                cert_name = re.sub(r'[^\w\-_.]', '_', cert_name)
                return f"certificate_{doc['id']}_{cert_name[:50]}.pdf"
        elif doc['type'] == "OTHER":
            return f"other_document_{doc['id']}.pdf"
        elif doc['type'] == "WITHHOLDING STATEMENT":
            return f"withholding_statement_{doc['id']}.pdf"
        elif doc['type'] == "DOCUSIGN":
            # Standalone DocuSign (rare — only if it appeared without a preceding form)
            return f"docusign_certificate_{doc['id']}.pdf"
        else:
            return f"{doc['type'].lower().replace(' ', '_')}_{doc['id']}.pdf"

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
        documents = self.group_pages(pdf_document, input_path=input_path, use_parallel=use_parallel, max_workers=max_workers)

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

    def get_llm_statistics(self, documents: List[Dict]) -> Dict:
        """
        Get statistics about LLM usage for a batch of documents.
        
        Returns:
            Dictionary with LLM usage statistics
        """
        total_pages = sum(len(doc['pages']) for doc in documents)
        llm_pages = sum(len(doc['pages']) for doc in documents if doc.get('method') == 'llm')
        logic_pages = total_pages - llm_pages
        
        # Calculate average confidence
        llm_docs = [doc for doc in documents if doc.get('method') == 'llm']
        avg_llm_confidence = sum(doc.get('confidence', 0) for doc in llm_docs) / len(llm_docs) if llm_docs else 0
        
        logic_docs = [doc for doc in documents if doc.get('method') == 'logic']
        avg_logic_confidence = sum(doc.get('confidence', 0) for doc in logic_docs) / len(logic_docs) if logic_docs else 0
        
        # Estimate tokens (rough estimate: 1 token ≈ 4 chars)
        total_chars = sum(len(doc.get('text', '')) for doc in documents)
        estimated_tokens = total_chars // 4
        
        return {
            'total_documents': len(documents),
            'total_pages': total_pages,
            'llm_pages': llm_pages,
            'logic_pages': logic_pages,
            'llm_percentage': (llm_pages / total_pages * 100) if total_pages > 0 else 0,
            'avg_llm_confidence': avg_llm_confidence,
            'avg_logic_confidence': avg_logic_confidence,
            'estimated_tokens': estimated_tokens,
            'estimated_cost_gpt4o_mini': estimated_tokens * 0.00000015  # $0.15 per 1M tokens
        }

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