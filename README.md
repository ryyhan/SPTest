# Smart PDF Splitter

A Streamlit-based application for automatically splitting a single PDF containing multiple mixed documents (e.g., Tax Forms, Certificates) into separate PDF files using AI-powered classification and OCR.

## ✨ Features

### Document Intelligence
- **Automatic Document Detection**: Identifies different form types (W-9, W-8BEN, W-8BEN-E, W-8EXP, W-8IMY) and separates them
- **🤖 AI-Powered Classification**: Use GPT-4o mini for intelligent document classification
- **👁️ LLM Vision OCR**: Extract text from scanned documents using GPT-4o vision models (most accurate)
- **Fuzzy Catalog Matching**: Detects form catalog numbers even with OCR errors (e.g., "25O47z" → "25047z")
- **Confidence Scoring**: Each detected form includes a confidence score (0-100%) to flag uncertain detections
- **Ambiguity Detection**: Warns when a page contains multiple form types

### OCR Engine Options
- **LLM Vision (GPT-4o)** ⭐: Most accurate, handles poor quality scans, requires API key
- **RapidOCR** (default): Fastest with excellent CPU performance
- **Tesseract**: Good for clean, standard forms
- **EasyOCR**: Best for skewed images or poor quality scans

### Performance
- **Parallel Processing**: Multi-threaded page analysis for faster processing
- **Smart Page Grouping**: Groups pages by form type with strict page count rules
- **Fallback Chain**: Automatically falls back to traditional OCR if LLM fails

### User Experience
- **Downloadable Results**: Provides download links for each extracted document
- **Quality Warnings**: Visual indicators for low confidence or ambiguous detections
- **Debug Mode**: Show confidence scores, matched patterns, and reasoning

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SPTest
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs all dependencies including LLM support. All features are available immediately.

### 4. Install Tesseract OCR (Required - Fallback)

Tesseract is required as a fallback OCR engine. Install it for your platform:

- **macOS**:
  ```bash
  brew install tesseract
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **Windows**: Download from [official Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Optional: Install EasyOCR

For better OCR on poor quality or skewed documents:

```bash
pip install easyocr
```

---

## 📖 Usage

### Running the Application

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`.

### Steps

1. **Configure Azure OpenAI** (one-time setup):
   
   Edit `llm_config.py` and add your Azure OpenAI credentials:
   
   ```python
   default_config = create_azure_config(
       api_key="your-api-key-here",
       azure_deployment="gpt-4o",
       azure_endpoint="https://your-resource.openai.azure.com/"
   )
   ```

2. **Upload PDF**: Upload a PDF file containing multiple mixed documents.

3. **Select OCR Engine**:
   - **LLM Vision (GPT-4o)** ⭐: Most accurate, best for poor quality scans
   - **RapidOCR** (recommended): Fastest with excellent accuracy
   - **Tesseract**: Good for clean, standard forms
   - **EasyOCR**: Best for poor quality or skewed images

4. **Advanced Options** (click to expand):
   - **Parallel Processing**: Enable multi-threaded analysis (faster)
   - **Worker Threads**: Adjust number of threads (default: 4)
   - **Debug Mode**: Show confidence scores and matched patterns
   - **🤖 AI-Powered Classification**: Enable LLM-based document classification

5. **Click "🚀 Start Processing"**: The PDF will be analyzed and split.

6. **View Results**:
   - Documents are listed with detected form types
   - ⚠️ warnings indicate low confidence or ambiguous detections
   - 🤖 icon indicates pages classified by LLM
   - Click **Download PDF** to save each document

---

## ⚙️ Advanced Configuration

### LLM Configuration File (`llm_config.py`)

All LLM settings including **prompts** are centralized in `llm_config.py` for easy customization.

**Quick Start:**

```python
from llm_config import create_azure_config

# Azure OpenAI Configuration
config = create_azure_config(
    api_key="your-azure-key-here",
    azure_deployment="gpt-4o",
    azure_endpoint="https://my-resource.openai.azure.com/",
    ocr_model="gpt-4o",              # Vision model for OCR
    classification_model="gpt-4o-mini"  # Text model for classification
)
```

**Configuration Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | None | Azure OpenAI API key (required) |
| `azure_deployment` | str | None | Azure deployment name (required) |
| `azure_endpoint` | str | None | Azure endpoint URL (required) |
| `azure_api_version` | str | "2024-02-15-preview" | Azure API version |
| `ocr_model` | str | "gpt-4o" | Vision model for OCR |
| `classification_model` | str | "gpt-4o-mini" | Text model for classification |
| `ocr_image_zoom` | float | 2.0 | Image zoom level (1.0-4.0) |

**Prompts Included:**
- `OCR_SYSTEM_PROMPT` - Prompt for LLM-based OCR text extraction
- `CLASSIFICATION_SYSTEM_PROMPT` - System prompt for document classification
- `CLASSIFICATION_USER_TEMPLATE` - User prompt template for classification

Edit these prompts in `llm_config.py` to customize LLM behavior.

**Using Custom Config in Code:**

```python
from DocSplitter import PDFSplitter
from llm_config import create_azure_config

# Create custom configuration
config = create_azure_config(
    api_key="your-key-here",
    azure_deployment="gpt-4o",
    azure_endpoint="https://my-resource.openai.azure.com/",
    ocr_image_zoom=3.0  # Higher quality OCR
)

# Initialize splitter with custom config
splitter = PDFSplitter(llm_config=config)

# Process PDF
documents = splitter.split_pdf("input.pdf", "output/")
```

---

## 🤖 AI Features

### 1. LLM Vision OCR

Extract text from scanned documents using GPT-4o vision models instead of traditional OCR.

**When to use:**
- ✅ Poor quality scans that traditional OCR struggles with
- ✅ Complex layouts with mixed text and form fields
- ✅ Maximum accuracy is required

**How it works:**
1. PDF page is converted to high-resolution image (2x zoom)
2. Image is sent to GPT-4o Vision API via Azure OpenAI
3. LLM extracts all text with context understanding
4. Falls back to traditional OCR if LLM fails

**Configuration:**
- Select **"LLM Vision"** as OCR engine in the dropdown
- Configure Azure OpenAI in `llm_config.py`
- Model: `gpt-4o` (vision-capable)

### 2. LLM Classification

Use GPT-4o mini to intelligently classify document types.

**When to use:**
- ✅ Complex or unusual forms
- ✅ Documents with mixed content
- ✅ OCR quality is poor (LLM can infer from context)

**How it works:**
1. Text is extracted from each PDF page (via OCR or native)
2. Full page text is sent to GPT-4o mini via Azure OpenAI
3. LLM analyzes content holistically and returns:
   - Document type (W-8BEN, W-8BEN-E, etc.)
   - Confidence score
   - Reasoning explanation
4. Falls back to rule-based detection if LLM fails

**Configuration:**
- Check **"Enable LLM classification"** in Advanced Options
- Configure Azure OpenAI in `llm_config.py`
- Model: `gpt-4o-mini` (cost-effective)

---

**💡 Key Difference:**
- **LLM Vision OCR** = Extracts text from images (replaces Tesseract/EasyOCR)
- **LLM Classification** = Identifies document type (replaces rule-based detection)

You can use either or both. Both require Azure OpenAI configuration.

---

## ☁️ Azure OpenAI Service

This application uses Azure OpenAI Service for all LLM features.

### Required Information

To use LLM features, you need:
1. **Azure OpenAI API Key**: From your Azure OpenAI resource
2. **Deployment Name**: Your GPT model deployment (e.g., `gpt-4o`, `gpt-4o-mini`)
3. **Endpoint URL**: Your Azure OpenAI endpoint (e.g., `https://my-resource.openai.azure.com/`)
4. **API Version**: Azure API version (default: `2024-02-15-preview`)

### Benefits

- ✅ Enterprise-grade security and compliance
- ✅ Dedicated capacity with provisioned throughput
- ✅ Data residency controls
- ✅ Private network connectivity

---

## 📋 Supported Form Types

| Form Type | Description | Pages | Catalog Number |
|-----------|-------------|-------|----------------|
| W-8BEN | Certificate of Foreign Status of Beneficial Owner (Individuals) | 1 | 25047Z |
| W-8BEN-E | Certificate of Status of Beneficial Owner (Entities) | 8 | 59689N |
| W-8EXP | Certificate of Foreign Government/Organization | 3 | 115(2) |
| W-8IMY | Certificate of Foreign Intermediary | 8 | 25402Q |
| W-9 | Request for Taxpayer Identification Number | 1 or 6 | 10231X |
| CERTIFICATE | Award certificates, completion certificates | Variable | N/A |
| OTHER | Unrecognized documents, instructions, cover letters | Variable | N/A |

---

## ⚙️ How It Works

### Processing Pipeline

```
┌─────────────────┐
│  Upload PDF     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract Pages   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Text Extraction (per page) │
│  ┌─────────────────────┐    │
│  │ 1. Native PDF Text  │    │
│  │ 2. LLM Vision OCR   │    │ ← If enabled & needed
│  │ 3. Traditional OCR  │    │ ← Fallback
│  └─────────────────────┘    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Document Classification    │
│  ┌─────────────────────┐    │
│  │ LLM Classification  │    │ ← If enabled
│  │ Rule-Based Fallback │    │
│  └─────────────────────┘    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Page Grouping              │
│  - Strict page count rules  │
│  - Form type boundaries     │
│  - Certificate separation   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  Output PDFs    │
└─────────────────┘
```

### Classification Tiers (Rule-Based Fallback)

When LLM classification is disabled, the system uses a 4-tier approach:

1. **Tier 0**: Unique catalog numbers (100% confidence, fuzzy-matched)
2. **Tier 1**: Full title pattern matching (95% confidence)
3. **Tier 2**: Form name detection (80% confidence)
4. **Tier 3**: Fuzzy string matching (60% confidence)

### Page Grouping Rules

Documents are split based on:
- **Form type changes**: Different types always split
- **Page count limits**: W-8BEN-E max 8 pages, forces split
- **Certificates**: Each certificate is a separate document
- **Start page detection**: New document if page claims to be "Page 1"

---

## 💰 Cost Estimates

### LLM Vision OCR (GPT-4o)
- ~$0.01-0.03 per page (image input + text output)
- 100 pages ≈ $1-3

### LLM Classification (GPT-4o mini)
- ~$0.00015 per page (text input only)
- 100 pages ≈ $0.015

### Combined (OCR + Classification)
- 100 pages ≈ $1-3 (most cost is vision OCR)

**Note:** Prices vary based on image size and text length. Azure OpenAI pricing may differ.

---

## 🛠️ Troubleshooting

### "No module named 'fitz'"
```bash
pip install PyMuPDF
```

### "Tesseract is not installed"
Install Tesseract OCR (see Installation step 4).

### "langchain-openai not installed"
```bash
pip install langchain-openai
```

### LLM features not working
1. Verify API key is correct
2. Check internet connectivity
3. For Azure: Verify deployment name and endpoint
4. Check API quota/limits in your OpenAI/Azure dashboard

### Low confidence warnings
- Try **LLM Vision OCR** for better text extraction
- Try a different traditional OCR engine (EasyOCR for poor quality)
- Check if the PDF is heavily compressed or skewed
- Enable LLM classification for better document detection

### Slow processing
- Use **RapidOCR** instead of EasyOCR or LLM Vision
- Enable **Parallel Processing** in Advanced Options
- Increase **Worker Threads** (if you have multiple CPU cores)
- Disable LLM features for faster rule-based processing

### LLM OCR produces poor results
- Ensure PDF pages are not extremely low resolution
- Try increasing image quality (currently 2x zoom)
- Some handwritten text may still be challenging
- Fall back to EasyOCR for heavily degraded documents

---

## 📁 Project Structure

```
SPTest/
├── app.py                  # Streamlit web application
├── DocSplitter.py          # Core PDF splitting logic with LLM support
├── llm_config.py           # LLM configuration (OpenAI/Azure settings)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── split_forms/           # Output directory for split PDFs
└── leagcyFiles/           # Legacy/prototype code (not used)
```

**Key Files:**

- **`llm_config.py`**: Centralized configuration for all LLM settings
  - Edit this file to customize OCR prompts, models, zoom levels, etc.
  - Supports both OpenAI and Azure OpenAI configurations
  - Create custom configs programmatically for different use cases

- **`DocSplitter.py`**: Main processing logic
  - Uses configuration from `llm_config.py`
  - Can be used directly in Python scripts with custom config

- **`app.py`**: Streamlit web interface
  - Reads configuration from UI or `llm_config.py` defaults

---

## 🔒 Security & Privacy

### Data Handling
- **Local Processing**: PDFs are processed locally, not uploaded to external servers
- **LLM API Calls**: Only page text/images are sent to OpenAI/Azure (not entire PDFs)
- **Temporary Files**: Processing uses temporary directories that are cleaned up

### Best Practices
- Use Azure OpenAI for enterprise compliance requirements
- Don't process sensitive documents with external LLM APIs
- Store API keys securely (use environment variables in production)
- Review your organization's data policies before enabling LLM features

---

## 📝 License

[Add your license here]

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
