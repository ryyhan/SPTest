# Smart PDF Splitter

A Streamlit-based application for automatically splitting a single PDF containing multiple mixed documents (e.g., Tax Forms, Certificates) into separate PDF files.

## Features

- **Automatic Document Detection**: Identifies different form types (W-9, W-8BEN, W-8BEN-E, W-8EXP, W-8IMY) and separates them.
- **Multi-Engine OCR Support**: 
  - **RapidOCR** (default, fastest, excellent accuracy on CPU)
  - **Tesseract** (fast, best for standard forms)
  - **EasyOCR** (better for skewed images/poor quality, slower)
- **Fuzzy Catalog Matching**: Detects form catalog numbers even with OCR errors (e.g., "25O47z" → "25047z").
- **Confidence Scoring**: Each detected form includes a confidence score (0-100%) to flag uncertain detections.
- **Ambiguity Detection**: Warns when a page contains multiple form types.
- **Parallel Processing**: Multi-threaded page analysis for faster processing.
- **Smart Page Grouping**: Groups pages by form type with configurable page count rules.
- **Downloadable Results**: Provides download links for each extracted document.

## Installation

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

### 4. Install Tesseract OCR (Required)

Tesseract is required for OCR functionality. Install it for your platform:

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

## Usage

### Running the Application

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`.

### Steps

1. **Upload PDF**: Upload a PDF file containing multiple mixed documents.

2. **Select OCR Engine**:
   - **RapidOCR** (recommended): Fastest with excellent accuracy
   - **Tesseract**: Good for clean, standard forms
   - **EasyOCR**: Best for poor quality or skewed images

3. **Advanced Options** (click to expand):
   - **Parallel Processing**: Enable multi-threaded analysis (faster)
   - **Worker Threads**: Adjust number of threads (default: 4)
   - **Debug Mode**: Show confidence scores and matched patterns

4. **View Results**:
   - Documents are listed with detected form types
   - ⚠️ warnings indicate low confidence or ambiguous detections
   - Click **Download PDF** to save each document

### Command Line Usage

You can also run the splitter directly from the command line:

```bash
python DocSplitter.py
```

Edit `DocSplitter.py` to change the input file and OCR engine.

## Supported Form Types

| Form Type | Description | Pages | Catalog Number |
|-----------|-------------|-------|----------------|
| W-8BEN | Foreign Status of Beneficial Owner (Individuals) | 1 | 25047Z |
| W-8BEN-E | Status of Beneficial Owner (Entities) | 8 | 59689N |
| W-8EXP | Foreign Government/Organization | 3 | 115(2) |
| W-8IMY | Foreign Intermediary | 8 | 25402Q |
| W-9 | Request for Taxpayer ID | 1 or 6 | 10231X |
| CERTIFICATE | Various certificates | Variable | N/A |
| OTHER | Unrecognized documents | Variable | N/A |

## Project Structure

```
SPTest/
├── app.py              # Streamlit web application
├── DocSplitter.py      # Core PDF splitting logic
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── split_forms/       # Output directory for split PDFs
└── leagcyFiles/       # Legacy/prototype code (not used)
```

## How It Works

1. **Text Extraction**: Extracts text from PDF pages using native PDF text or OCR.
2. **Form Detection** (4-tier approach):
   - **Tier 0**: Unique catalog numbers (100% confidence, fuzzy-matched)
   - **Tier 1**: Full title pattern matching (95% confidence)
   - **Tier 2**: Form name detection (80% confidence)
   - **Tier 3**: Fuzzy string matching (60% confidence)
3. **Page Grouping**: Groups pages by form type using page count rules.
4. **Output**: Creates separate PDF files for each detected document.

## Troubleshooting

### "No module named 'fitz'"
```bash
pip install PyMuPDF
```

### "Tesseract is not installed"
Install Tesseract OCR (see Installation step 4).

### Low confidence warnings
- Try a different OCR engine (EasyOCR for poor quality scans)
- Check if the PDF is heavily compressed or skewed
- Manually review flagged documents

### Slow processing
- Enable **Parallel Processing** in Advanced Options
- Increase **Worker Threads** (if you have multiple CPU cores)
- Use RapidOCR instead of EasyOCR

## License

[Add your license here]
