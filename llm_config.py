"""
LLM Configuration for Smart PDF Splitter

Azure OpenAI configuration for LangChain clients.
"""

from typing import Optional
from dataclasses import dataclass


# ===========================================
# Prompts
# ===========================================

OCR_SYSTEM_PROMPT = """You are an expert OCR system. Extract ALL text from this document image with high accuracy.

IMPORTANT:
- Preserve the exact text content, including form fields, labels, values, and footer text
- Maintain logical line breaks and structure where appropriate
- Include catalog numbers, form IDs, and any small print
- If you see "Page X of Y" or similar, include it
- Do not add any commentary - return ONLY the extracted text

Extracted text:"""


CLASSIFICATION_SYSTEM_PROMPT = """You are an expert document classification assistant specialized in US tax forms and financial documents. Your task is to analyze the COMPLETE text from a PDF page and determine what type of document this page belongs to.

AVAILABLE DOCUMENT TYPES:
1. W-8BEN: Certificate of Foreign Status of Beneficial Owner (Individuals) - must have form fields
2. W-8BEN-E: Certificate of Status of Beneficial Owner (Entities) - must have form fields
3. W-8EXP: Certificate of Foreign Government or Organization - must have form fields
4. W-8IMY: Certificate of Foreign Intermediary - must have form fields
5. W-9: Request for Taxpayer Identification Number - must have form fields
6. CERTIFICATE: Award certificates, completion certificates, certification letters (no form fields)
7. WITHHOLDING STATEMENT: Any withholding statement, tax notification, or payment advice
8. OTHER: Everything else - instructions, cover letters, supporting documents

CRITICAL CLASSIFICATION RULES:
1. WITHHOLDING STATEMENTS: Any page that is a withholding statement, tax notification, or payment advice should be classified as WITHHOLDING STATEMENT, EVEN IF it mentions being "attached to" or "part of" a W-8 form
2. Forms MUST have fillable fields: A page is ONLY a W-8 form if it has blank lines to fill, checkboxes, signature blocks with "Sign Here" instructions
3. Prose text = NOT a form: Pages with paragraphs of text, even if titled "Statement" or "Certificate", are NOT tax forms (unless they are a WITHHOLDING STATEMENT)
4. "Part of" doesn't mean same type: A withholding statement that says "This is part of Form W-8IMY" should be classified as WITHHOLDING STATEMENT, not W-8IMY
5. Each page type is a separate document: Withholding statements, forms, and certificates should each be their own document type
6. Look for form structure: Actual forms have: "☐" checkboxes, "________" fill lines, "Signature" blocks, "Date" fields

EXAMPLES:
- "Withholding Statement" + mentions W-8IMY + no form fields → WITHHOLDING STATEMENT (NOT W-8IMY)
- "This statement is attached to Form W-8IMY" → OTHER (the statement itself is not the form)
- "Notification of withholding for W-8BEN" → WITHHOLDING STATEMENT (NOT W-8BEN)
- "Form W-8IMY" + fillable fields + signature → W-8IMY
- "Beneficial Owner Statement" + prose text → OTHER or CERTIFICATE
- "We certify..." + no form fields → CERTIFICATE
- Dense legal text about tax treaties → OTHER (instructions)

You will receive the FULL text from one page. Analyze it holistically and return your classification as JSON.

REMEMBER: When in doubt, classify prose/text-heavy pages as OTHER. Only classify as W-8 forms if the page has actual form fields to fill out."""


CLASSIFICATION_USER_TEMPLATE = """Analyze this COMPLETE text from page {page_num} of a PDF and classify what type of document this page is.

FULL PAGE TEXT:
---
{text}
---

Respond with ONLY a valid JSON object in this exact format:
{{
    "document_type": "W-8BEN",
    "confidence": 0.95,
    "reasoning": "Explain: Does this have form fields (checkboxes, fill lines, signature)? Or is it prose text? If it mentions a form but has no fields, classify as OTHER.",
    "is_first_page": false,
    "form_structure_detected": true,
    "mentions_other_forms": ["W-8BEN-E"] if any, otherwise omit
}}

document_type must be one of: W-8BEN, W-8BEN-E, W-8EXP, W-8IMY, W-9, CERTIFICATE, WITHHOLDING STATEMENT, OTHER
confidence should be between 0.0 and 1.0

is_first_page rules (CRITICAL - read carefully):
- Set to TRUE if this is a single-page document OR the first page of a multi-page document (look for "Page 1 of X", form title, etc.)
- Set to FALSE if this is a continuation page (Page 2+, or if the page says "Part II", "Section B", etc.)
- When in doubt, set to FALSE

form_structure_detected should be true ONLY if the page has form fields, checkboxes, signature lines (NOT just prose text)

CRITICAL:
- Withholding statements = WITHHOLDING STATEMENT (even if they say "part of Form W-8...")
- Tax notifications = WITHHOLDING STATEMENT
- Beneficial owner statements without form fields = OTHER
- ONLY pages with fillable form fields = W-8 forms
- is_first_page = TRUE if it is a single-page document OR the first page of a multi-page document"""


# ===========================================
# Configuration
# ===========================================

@dataclass
class LLMConfig:
    """
    Azure OpenAI configuration for LLM-based OCR and Classification.
    """
    
    # ===========================================
    # Required: Azure OpenAI Credentials
    # ===========================================
    
    api_key: str
    
    azure_deployment: str
    azure_endpoint: str
    azure_api_version: str = "2024-02-15-preview"
    
    # ===========================================
    # Optional: Model Selection
    # ===========================================
    
    # Model for OCR (vision-capable)
    ocr_model: str = "gpt-4o"
    
    # Model for document classification (text-based)
    classification_model: str = "gpt-4o-mini"
    
    # ===========================================
    # Optional: OCR Quality
    # ===========================================
    
    # Image zoom level (1.0 = normal, 2.0 = 2x, etc.)
    ocr_image_zoom: float = 2.0
    
    # ===========================================
    # Helper Methods
    # ===========================================
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate the configuration.
        Returns: (is_valid, error_message)
        """
        if not self.api_key:
            return False, "API key is required"
        
        if not self.azure_deployment:
            return False, "Azure deployment name is required"
        
        if not self.azure_endpoint:
            return False, "Azure endpoint is required"
        
        return True, ""


# ===========================================
# Convenience Function
# ===========================================

def create_azure_config(
    api_key: str,
    azure_deployment: str,
    azure_endpoint: str,
    azure_api_version: str = "2024-02-15-preview",
    ocr_model: str = "gpt-4o",
    classification_model: str = "gpt-4o-mini",
    ocr_image_zoom: float = 2.0
) -> LLMConfig:
    """
    Create configuration for Azure OpenAI.
    
    Args:
        api_key: Your Azure OpenAI API key
        azure_deployment: Azure deployment name
        azure_endpoint: Azure endpoint URL
        azure_api_version: Azure API version
        ocr_model: Vision model for OCR
        classification_model: Text model for classification
        ocr_image_zoom: Image zoom level
    
    Returns:
        LLMConfig instance
    """
    return LLMConfig(
        api_key=api_key,
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        ocr_model=ocr_model,
        classification_model=classification_model,
        ocr_image_zoom=ocr_image_zoom
    )


# ===========================================
# Default Configuration (EDIT THIS!)
# ===========================================

# Configure your Azure OpenAI credentials here
# This configuration is used by the UI automatically
default_config = create_azure_config(
    api_key="",  # ← Add your Azure OpenAI API key here
    azure_deployment="",  # ← Add your deployment name here (e.g., "gpt-4o")
    azure_endpoint="",  # ← Add your endpoint here (e.g., "https://your-resource.openai.azure.com/")
    azure_api_version="2024-02-15-preview",
    ocr_model="gpt-4o",
    classification_model="gpt-4o-mini",
    ocr_image_zoom=2.0
)
