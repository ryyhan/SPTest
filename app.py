import streamlit as st
import os
import tempfile
from DocSplitter import PDFSplitter

# Set page configuration
st.set_page_config(
    page_title="Smart PDF Splitter",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 Smart PDF Splitter")
    st.markdown("""
    ### Intelligent Document Separation
    Upload a single PDF containing multiple mixed documents (Tax Forms, Certificates, etc.).
    This tool will automatically identify and split them into separate PDF files.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # OCR Engine selection
    engine_choice = st.selectbox(
        "Select OCR Engine for Scanned Pages",
        [
            "RapidOCR (Fastest, excellent accuracy on CPU)",
            "Tesseract (Fast, best for standard forms)", 
            "EasyOCR (Better for skewed images/poor quality, but slow)"
        ]
    )
    
    if "RapidOCR" in engine_choice:
        ocr_engine = "rapidocr"
    elif "EasyOCR" in engine_choice:
        ocr_engine = "easyocr"
    else:
        ocr_engine = "tesseract"

    if uploaded_file is not None:
        st.divider()
        st.subheader("Processing Results")
        
        # Create a progress container
        status_container = st.empty()
        status_container.info("Initializing processor...")
        
        # Create a temporary directory for processing
        # We use a temporary directory so we don't clutter the server/local storage
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save uploaded file to temp path
                input_path = os.path.join(temp_dir, "input.pdf")
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_container.info("Analyzing and splitting PDF... This may take a moment (OCR active).")
                
                # Initialize splitter
                splitter = PDFSplitter(ocr_engine=ocr_engine)
                
                # Create output directory (persistent)
                output_dir = "split_forms"
                os.makedirs(output_dir, exist_ok=True)
                
                # Run the split logic
                documents = splitter.split_pdf(input_path, output_dir)
                
                status_container.success(f"✅ Successfully split into {len(documents)} document(s)!")
                
                # Display results in a grid or list
                if documents:
                    st.markdown(f"**Found {len(documents)} documents:**")
                    
                    for i, doc in enumerate(documents):
                        # Generate the filename again to locate the file
                        filename = splitter.generate_filename(doc)
                        filepath = os.path.join(output_dir, filename)
                        
                        # Create a card-like layout for each document
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{i+1}. {doc['type']}**")
                                st.caption(f"Pages: {len(doc['pages'])} | ID: {doc['id']}")
                                
                            with col2:
                                # Read the file for the download button
                                with open(filepath, "rb") as f:
                                    st.download_button(
                                        label="⬇️ Download PDF",
                                        data=f,
                                        file_name=filename,
                                        mime="application/pdf",
                                        key=f"download_{i}"
                                    )
                            st.divider()
                            
            except Exception as e:
                status_container.error(f"An error occurred during processing: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
