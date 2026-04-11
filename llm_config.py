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
1. W-8BEN: Certificate of Foreign Status of Beneficial Owner (Individuals)
2. W-8BEN-E: Certificate of Status of Beneficial Owner (Entities)
3. W-8EXP: Certificate of Foreign Government or Organization
4. W-8IMY: Certificate of Foreign Intermediary
5. W-9: Request for Taxpayer Identification Number
6. CERTIFICATE: Award certificates, completion certificates, certification letters (standalone, not part of a tax form)
7. WITHHOLDING STATEMENT: **Only** documents explicitly serving as a Withholding Statement (containing specific withholding allocations/breakdowns for payees).
8. OTHER: Everything else - instructions pages NOT part of an identified tax form, cover letters, supporting documents, generic tax notifications, payment advices, general correspondence.

CRITICAL CLASSIFICATION RULES (in order of priority — Rule 0 overrides all others):

**RULE 0 — FORM HEADER/FOOTER STAMP IS ABSOLUTE TRUTH (HIGHEST PRIORITY):**
If the page contains a form name in its header, footer, or watermark — such as:
  - "Form W-9 (Rev. 11-2017) Page 5"
  - "Form W-8IMY (Rev. 10-2021) Page 3"
  - "W-8BEN-E (Rev. July 2017) Page 2"
  - "Form W-8EXP Page 2 of 3"
Then classify the page as THAT form type, regardless of whether the content is prose, instructions, or has no form fields.
This rule applies to ALL pages of multi-page tax forms — the IRS prints the form name on EVERY page including instruction pages.
Do NOT classify these as OTHER just because the content is instructional prose.
is_first_page should be FALSE for any page stamped with "Page 2", "Page 3", etc.

1. STRICT WITHHOLDING STATEMENTS: ONLY classify a page as WITHHOLDING STATEMENT if it acts as the actual withholding breakdown/allocation document. General tax instructions, general manuals, generic tax notifications, and payment advices that merely *mention* withholding should be classified as OTHER.
2. Form fields are NOT required on every page: Multi-page forms (W-9, W-8BEN-E, W-8IMY, W-8EXP) include instruction pages with dense prose — these are still part of the form if the header/footer identifies them as such.
3. Prose text alone = OTHER only when there is NO form header identifying it: Pages with paragraphs of text and no form identification header are OTHER (unless they are a true WITHHOLDING STATEMENT).
4. "Part of" doesn't mean same type: A true withholding statement that says "This is part of Form W-8IMY" should be classified as WITHHOLDING STATEMENT, not W-8IMY.
5. Each document type is separate: Withholding statements, forms, and certificates should each be their own document type.
6. Standalone first pages of forms still need form structure: If a page claims to be Page 1 of a form, it should have form fields (checkboxes, fill lines, signature blocks). If it has NO form fields and NO continuation page marker, it may not be the form itself.

EXAMPLES:
- "Form W-9 (Rev. 11-2017) Page 5" + instructions prose → W-9, is_first_page=FALSE  ← RULE 0 applies
- "Form W-8IMY (Rev. 10-2021) Page 3" + instructions prose → W-8IMY, is_first_page=FALSE  ← RULE 0 applies
- "Form W-8BEN-E Page 2 of 8" + continuation content → W-8BEN-E, is_first_page=FALSE  ← RULE 0 applies
- "Withholding Statement" + mentions W-8IMY + has allocation breakdowns → WITHHOLDING STATEMENT
- "This statement is attached to Form W-8IMY" (no form stamp, no fields) → OTHER
- "Notification of withholding for W-8BEN" → OTHER (generic notification)
- Payment Advice displaying taxes paid → OTHER
- "Form W-8IMY" + fillable fields + signature (Page 1) → W-8IMY, is_first_page=TRUE
- "We certify..." + no form fields + no form stamp → CERTIFICATE
- Dense legal text about tax treaties with no form stamp → OTHER (instructions)

You will receive the FULL text from one page. Analyze it holistically and return your classification as JSON.

REMEMBER: If you can see the form name stamped anywhere on the page (header, footer, watermark), ALWAYS classify it as that form — even for instruction pages. Never classify a form-stamped page as OTHER."""



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
- Set to TRUE if this is Page 1 of a form (look for "Page 1 of X", form title at top with fields below, etc.)
- Set to FALSE if this is a continuation page — "Page 2", "Page 3", "Part II", "Section B", footer says "Page X" where X > 1
- If the page header/footer says "Form W-9 Page 5" → is_first_page = FALSE
- When in doubt, set to FALSE

form_structure_detected should be true ONLY if the page has form fields, checkboxes, signature lines.
For continuation/instruction pages of a form, form_structure_detected = false is correct and expected.

CRITICAL PRIORITY REMINDER:
- FIRST: Check if there is a form name stamped in the header/footer (e.g. "Form W-9 Page 5") → if yes, classify as THAT form
- SECOND: For pages with no form stamp — Withholding statements = WITHHOLDING STATEMENT (only if it is the actual allocation statement)
- Generic tax notifications and payment advices = OTHER
- Instructions pages with NO form stamp = OTHER
- Beneficial owner statements without form fields and no form stamp = OTHER
- is_first_page = TRUE only for the actual first page of a form"""


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
    ocr_image_zoom=3.0
)
