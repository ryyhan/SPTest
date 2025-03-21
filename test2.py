import os
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfWriter

# Define the form titles and other keywords
FORM_TITLES = ["Form W-9", "Form 8-EXP", "Form W-8BEN", "Form W-8BEN-E", "Form W-8IMY"]
OTHER_KEYWORDS = ["certificate"]

def extract_text_from_page(page):
    """Extract text from a PDF page using pdfplumber."""
    text = page.extract_text()
    if not text:  # If no text is found, try OCR
        image = page.to_image(resolution=300)
        text = pytesseract.image_to_string(image.original)
    return text

def split_pdf(input_pdf_path, output_dir):
    """Split the input PDF into multiple sub-PDFs based on form titles and other keywords."""
    # Open the PDF file
    pdf_document = fitz.open(input_pdf_path)
    
    # Initialize dictionaries to store pages for each form and other categories
    form_pages = {title: [] for title in FORM_TITLES}
    other_keywords_pages = {keyword: [] for keyword in OTHER_KEYWORDS}
    other_documents_pages = []
    
    # Process each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = extract_text_from_page(page)
        
        # Check if the page matches any form title
        matched_form = False
        for form_title in FORM_TITLES:
            if text.startswith(form_title):
                form_pages[form_title].append(page_num)
                matched_form = True
                break
        
        # If no form title matches, check for other keywords
        if not matched_form:
            matched_keyword = False
            for keyword in OTHER_KEYWORDS:
                if keyword.lower() in text.lower():
                    other_keywords_pages[keyword].append(page_num)
                    matched_keyword = True
                    break
            
            # If no keyword matches, add to "Other Documents"
            if not matched_keyword:
                other_documents_pages.append(page_num)
    
    # Save each form's pages as a separate PDF
    for form_title, pages in form_pages.items():
        if pages:
            output_pdf_path = os.path.join(output_dir, f"{form_title.replace(' ', '_')}.pdf")
            save_pages_as_pdf(input_pdf_path, pages, output_pdf_path)
    
    # Save pages with other keywords as separate PDFs
    for keyword, pages in other_keywords_pages.items():
        if pages:
            output_pdf_path = os.path.join(output_dir, f"{keyword.replace(' ', '_')}.pdf")
            save_pages_as_pdf(input_pdf_path, pages, output_pdf_path)
    
    # Save remaining pages as "Other Documents" PDF
    if other_documents_pages:
        output_pdf_path = os.path.join(output_dir, "Other_Documents.pdf")
        save_pages_as_pdf(input_pdf_path, other_documents_pages, output_pdf_path)

def save_pages_as_pdf(input_pdf_path, pages, output_pdf_path):
    """Save specific pages from the input PDF to a new PDF file."""
    with open(input_pdf_path, "rb") as input_pdf_file:
        reader = PyPDF2.PdfReader(input_pdf_file)
        writer = PdfWriter()
        
        for page_num in pages:
            writer.add_page(reader.pages[page_num])
        
        with open(output_pdf_path, "wb") as output_pdf_file:
            writer.write(output_pdf_file)

# usage
input_pdf_path = "form-image-other.pdf"
output_dir = "output_pdfs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

split_pdf(input_pdf_path, output_dir)