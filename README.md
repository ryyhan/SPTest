# Smart PDF Splitter

A Streamlit-based application for automatically splitting a single PDF containing multiple mixed documents (e.g., Tax Forms, Certificates) into separate PDF files.

## Features
- **Automatic Document Detection**: Identifies different form types (W-9, W-8BEN, etc.) and separates them.
- **OCR Support**: Uses Tesseract (default) or EasyOCR (optional) to extract text from scanned pages.
- **Smart Page Grouping**: Groups pages belonging to the same document based on form rules.
- **Downloadable Results**: Provides download links for each extracted document.

## Installation

1.  **Clone the repository** (or download the source code).

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    pip install -r requirements.txt
    pip install easyocr numpy
    pip install opencv-python
    pip install thefuzz python-Levenshtein

    ```

3.  **Install Tesseract OCR** (Required for OCR):
    -   **macOS**: `brew install tesseract`
    -   **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
    -   **Windows**: Download from the [official Tesseract GitHub repository](https://github.com/UB-Mannheim/tesseract/wiki).

## Usage

1.  **Run the application**:

    ```bash
    streamlit run app.py
    ```

2.  **Upload PDF**:
    -   Open the web interface (usually `http://localhost:8501`).
    -   Upload a PDF file containing multiple documents.

3.  **Select OCR Engine**:
    -   Choose **Tesseract** (faster, good for clean text) or **EasyOCR** (better for poor quality/skewed images).

4.  **View Results**:
    -   The app will process the PDF and display the detected documents.
    -   Click the **Download PDF** button next to each document to save it.

## Project Structure
-   `app.py`: The main Streamlit application entry point.
-   `DocSplitter.py`: The core logic for PDF parsing, text extraction, and document splitting.
-   `requirements.txt`: Lists all Python dependencies.
