"""
LLM Configuration for Smart PDF Splitter

Centralized configuration for LangChain OpenAI and Azure OpenAI clients.
Modify these settings to customize LLM behavior for OCR and classification.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """
    Configuration for LLM-based features (OCR and Classification).
    
    Supports both OpenAI and Azure OpenAI Service.
    """
    
    # ===========================================
    # API Configuration
    # ===========================================
    
    # API Key (required for LLM features)
    api_key: Optional[str] = None
    
    # API Base URL (optional, for custom endpoints)
    # Default for OpenAI: https://api.openai.com/v1
    api_base: Optional[str] = None
    
    # ===========================================
    # Azure OpenAI Configuration (optional)
    # ===========================================
    
    # Set to True to use Azure OpenAI instead of OpenAI
    use_azure: bool = False
    
    # Azure deployment name (required for Azure)
    # Example: "gpt-4o", "gpt-4o-mini"
    azure_deployment: Optional[str] = None
    
    # Azure endpoint URL (required for Azure)
    # Example: "https://my-resource.openai.azure.com/"
    azure_endpoint: Optional[str] = None
    
    # Azure API version (required for Azure)
    # Common versions: "2024-02-15-preview", "2024-03-01-preview"
    azure_api_version: str = "2024-02-15-preview"
    
    # ===========================================
    # LLM OCR Configuration (Vision Models)
    # ===========================================
    
    # Enable LLM-based OCR using vision models
    use_llm_ocr: bool = False
    
    # Model for OCR (vision-capable model required)
    # Recommended: "gpt-4o" (best quality)
    # Alternative: "gpt-4-turbo" (older, less accurate)
    ocr_model: str = "gpt-4o"
    
    # Max tokens for OCR response
    # Higher = more detailed text extraction, but slower and more expensive
    ocr_max_tokens: int = 2000
    
    # Temperature for OCR (keep low for consistency)
    ocr_temperature: float = 0.0
    
    # Image detail level for OCR
    # Options: "low", "high", "auto"
    # "high" = better quality but more tokens
    # "low" = faster and cheaper
    ocr_image_detail: str = "high"
    
    # Image zoom level for OCR (PyMuPDF matrix)
    # Higher = better quality but larger images
    # Recommended: 2.0 (2x zoom)
    ocr_image_zoom: float = 2.0
    
    # OCR prompt customization
    ocr_system_prompt: str = """You are an expert OCR system specialized in document text extraction.
Your task is to extract ALL text from document images with maximum accuracy.

IMPORTANT GUIDELINES:
1. Extract every visible word, including:
   - Form titles and headers
   - Field labels and values
   - Footer text and page numbers
   - Catalog numbers and form IDs
   - Fine print and disclaimers

2. Preserve structure:
   - Maintain logical line breaks
   - Keep related text together
   - Indicate section breaks clearly

3. Be precise:
   - Don't add commentary or explanations
   - Don't skip text that appears unclear
   - Mark uncertain text with [?] if needed

4. Special attention to:
   - Catalog numbers (e.g., "25047Z", "59689N")
   - Form names (e.g., "W-8BEN-E", "Form W-9")
   - Page indicators (e.g., "Page 1 of 8")
   - Revision dates (e.g., "Rev. 10-2018")

Return ONLY the extracted text. Do not add any preamble or conclusion."""

    ocr_user_prompt_template: str = """Extract all text from this document image.

{image}

Extracted text:"""
    
    # ===========================================
    # LLM Classification Configuration
    # ===========================================
    
    # Enable LLM-based document classification
    use_llm_classification: bool = False
    
    # Model for classification (text-based)
    # Recommended: "gpt-4o-mini" (cost-effective)
    # Alternative: "gpt-4o" (more accurate), "gpt-4-turbo"
    classification_model: str = "gpt-4o-mini"
    
    # Max tokens for classification response
    classification_max_tokens: int = 400
    
    # Temperature for classification (keep low for consistency)
    classification_temperature: float = 0.1
    
    # Timeout for classification API calls (seconds)
    classification_timeout: int = 60
    
    # ===========================================
    # Retry Configuration
    # ===========================================
    
    # Number of retry attempts for failed API calls
    max_retries: int = 3
    
    # Delay between retries (seconds)
    retry_delay: float = 1.0
    
    # ===========================================
    # Supported Document Types
    # ===========================================
    
    # List of valid document types for classification
    valid_document_types: list = field(default_factory=lambda: [
        "W-8BEN",
        "W-8BEN-E",
        "W-8EXP",
        "W-8IMY",
        "W-9",
        "CERTIFICATE",
        "OTHER"
    ])
    
    # ===========================================
    # Logging & Debugging
    # ===========================================
    
    # Enable verbose logging
    verbose: bool = True
    
    # Log API requests and responses
    log_api_calls: bool = False
    
    # ===========================================
    # Helper Methods
    # ===========================================
    
    def get_ocr_model(self) -> str:
        """Get the model name for OCR."""
        return self.ocr_model
    
    def get_classification_model(self) -> str:
        """Get the model name for classification."""
        return self.classification_model
    
    def is_azure(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return self.use_azure and self.azure_deployment and self.azure_endpoint
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate the configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.use_llm_ocr or self.use_llm_classification:
            if not self.api_key:
                return False, "API key is required for LLM features"
            
            if self.use_azure:
                if not self.azure_deployment:
                    return False, "Azure deployment name is required"
                if not self.azure_endpoint:
                    return False, "Azure endpoint is required"
        
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'api_key': self.api_key,
            'api_base': self.api_base,
            'use_azure': self.use_azure,
            'azure_deployment': self.azure_deployment,
            'azure_endpoint': self.azure_endpoint,
            'azure_api_version': self.azure_api_version,
            'use_llm_ocr': self.use_llm_ocr,
            'ocr_model': self.ocr_model,
            'ocr_max_tokens': self.ocr_max_tokens,
            'ocr_temperature': self.ocr_temperature,
            'ocr_image_detail': self.ocr_image_detail,
            'ocr_image_zoom': self.ocr_image_zoom,
            'use_llm_classification': self.use_llm_classification,
            'classification_model': self.classification_model,
            'classification_max_tokens': self.classification_max_tokens,
            'classification_temperature': self.classification_temperature,
            'classification_timeout': self.classification_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'verbose': self.verbose,
            'log_api_calls': self.log_api_calls
        }


# ===========================================
# Default Configuration Instance
# ===========================================

# Create a default configuration that can be imported and modified
default_config = LLMConfig()

# ===========================================
# Quick Configuration Examples
# ===========================================

def create_openai_config(
    api_key: str,
    ocr_model: str = "gpt-4o",
    classification_model: str = "gpt-4o-mini",
    **kwargs
) -> LLMConfig:
    """
    Create a configuration for OpenAI API.
    
    Args:
        api_key: Your OpenAI API key
        ocr_model: Model for OCR (vision)
        classification_model: Model for classification (text)
        **kwargs: Additional LLMConfig parameters
    
    Returns:
        Configured LLMConfig instance
    """
    return LLMConfig(
        api_key=api_key,
        use_azure=False,
        ocr_model=ocr_model,
        classification_model=classification_model,
        **kwargs
    )


def create_azure_config(
    api_key: str,
    azure_deployment: str,
    azure_endpoint: str,
    azure_api_version: str = "2024-02-15-preview",
    ocr_model: str = "gpt-4o",
    classification_model: str = "gpt-4o-mini",
    **kwargs
) -> LLMConfig:
    """
    Create a configuration for Azure OpenAI Service.
    
    Args:
        api_key: Your Azure OpenAI API key
        azure_deployment: Azure deployment name
        azure_endpoint: Azure endpoint URL
        azure_api_version: Azure API version
        ocr_model: Model for OCR (vision)
        classification_model: Model for classification (text)
        **kwargs: Additional LLMConfig parameters
    
    Returns:
        Configured LLMConfig instance
    """
    return LLMConfig(
        api_key=api_key,
        use_azure=True,
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        ocr_model=ocr_model,
        classification_model=classification_model,
        **kwargs
    )


# ===========================================
# Usage Examples
# ===========================================

if __name__ == "__main__":
    # Example 1: Basic OpenAI configuration
    config1 = create_openai_config(
        api_key="sk-your-openai-key-here"
    )
    print("OpenAI Config:", config1.to_dict())
    
    # Example 2: Azure OpenAI configuration
    config2 = create_azure_config(
        api_key="your-azure-key-here",
        azure_deployment="gpt-4o",
        azure_endpoint="https://my-resource.openai.azure.com/"
    )
    print("Azure Config:", config2.to_dict())
    
    # Example 3: Custom configuration
    config3 = LLMConfig(
        api_key="sk-your-key-here",
        use_llm_ocr=True,
        use_llm_classification=True,
        ocr_model="gpt-4o",
        classification_model="gpt-4o-mini",
        ocr_image_zoom=2.0,
        verbose=True
    )
    print("Custom Config:", config3.to_dict())
    
    # Example 4: Validate configuration
    is_valid, error = config3.validate()
    print(f"Valid: {is_valid}, Error: {error}")
