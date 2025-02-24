import os
import pdfplumber
from PyPDF2 import PdfReader, PdfWriter

# Function to check if a page contains a form
def is_form_page(text):
    # List of form keywords to search for
    form_keywords = ["Form W-9", "Form 8-EXP", "Form W-8BEN", "Form W-8BEN-E", "Form W-8IMY"]
    return any(keyword in text for keyword in form_keywords)

# Function to check if a page contains an image
def has_image(page):
    # Extract images from the page
    images = page.images
    return len(images) > 0

# Main function to split the PDF
def split_pdf(input_pdf_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize PDF writers for each category
    form_writer = PdfWriter()
    image_writer = PdfWriter()
    other_writer = PdfWriter()

    # Open the input PDF
    with pdfplumber.open(input_pdf_path) as pdf:
        reader = PdfReader(input_pdf_path)
        
        for i, page in enumerate(pdf.pages):
            # Extract text from the page
            text = page.extract_text()
            
            # Check if the page is a form
            if is_form_page(text):
                form_writer.add_page(reader.pages[i])
            # Check if the page contains an image
            elif has_image(page):
                image_writer.add_page(reader.pages[i])
            # Otherwise, classify as "other"
            else:
                other_writer.add_page(reader.pages[i])

    # Write the classified pages to separate PDFs
    form_output_path = os.path.join(output_dir, "forms.pdf")
    image_output_path = os.path.join(output_dir, "images.pdf")
    other_output_path = os.path.join(output_dir, "other_documents.pdf")

    with open(form_output_path, "wb") as form_file:
        form_writer.write(form_file)

    with open(image_output_path, "wb") as image_file:
        image_writer.write(image_file)

    with open(other_output_path, "wb") as other_file:
        other_writer.write(other_file)

    print(f"PDFs successfully split and saved in {output_dir}")

# Example usage
input_pdf_path = "other-form-form.pdf"  # Path to your input PDF
output_directory = "output_pdfs"  # Directory where output PDFs will be saved

split_pdf(input_pdf_path, output_directory)