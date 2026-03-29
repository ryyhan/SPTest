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
            "EasyOCR (Better for skewed images/poor quality, but slow)",
            "LLM Vision (GPT-4o - Most accurate, requires API key, slower)"
        ]
    )

    # Advanced options expander
    with st.expander("⚙️ Advanced Options"):
        use_parallel = st.checkbox("Enable parallel processing (faster)", value=True)
        max_workers = st.slider("Number of worker threads", 1, 8, 4)
        show_debug = st.checkbox("Show debug information", value=False)

        st.divider()

        # LLM Classification Option
        st.subheader("🤖 AI-Powered Classification")
        use_llm = st.checkbox(
            "Enable LLM classification (higher accuracy, slower)",
            value=False,
            help="Use GPT-4o mini to classify pages. More accurate but slower than rule-based detection."
        )
        
        # LLM OCR Option
        use_llm_ocr = "LLM Vision" in engine_choice
        
        # Show API configuration if either LLM feature is enabled
        if use_llm or use_llm_ocr:
            api_key = st.text_input(
                "API Key",
                type="password",
                help="Your OpenAI API key or your company's LLM API key"
            )
            api_base = st.text_input(
                "API Base URL (optional)",
                placeholder="https://api.openai.com/v1",
                help="Custom API endpoint (e.g., your company's LLM gateway)"
            )
            
            # Azure OpenAI Configuration
            st.markdown("**Azure OpenAI Configuration** (optional)")
            use_azure = st.checkbox(
                "Use Azure OpenAI",
                value=False,
                help="Enable if you're using Azure OpenAI Service"
            )
            
            if use_azure:
                azure_deployment = st.text_input(
                    "Azure Deployment Name",
                    placeholder="gpt-4o",
                    help="Your Azure OpenAI deployment name"
                )
                azure_endpoint = st.text_input(
                    "Azure Endpoint",
                    placeholder="https://your-resource.openai.azure.com/",
                    help="Your Azure OpenAI endpoint URL"
                )
                azure_api_version = st.text_input(
                    "Azure API Version",
                    value="2024-02-15-preview",
                    help="Azure API version"
                )
            else:
                azure_deployment = None
                azure_endpoint = None
                azure_api_version = None
            
            llm_model = st.text_input(
                "Model Name",
                value="gpt-4o",
                help="The LLM model to use (e.g., gpt-4o for OCR, gpt-4o-mini for classification)"
            )
        else:
            api_key = None
            api_base = None
            llm_model = "gpt-4o"
            azure_deployment = None
            azure_endpoint = None
            azure_api_version = None

    if "RapidOCR" in engine_choice:
        ocr_engine = "rapidocr"
    elif "EasyOCR" in engine_choice:
        ocr_engine = "easyocr"
    elif "LLM Vision" in engine_choice:
        ocr_engine = "llm"
    else:
        ocr_engine = "tesseract"

    if uploaded_file is not None:
        st.divider()
        st.subheader("Processing Results")

        # Create a progress container
        status_container = st.empty()
        status_container.info("Initializing processor...")

        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save uploaded file to temp path
                input_path = os.path.join(temp_dir, "input.pdf")
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                status_container.info("Analyzing and splitting PDF... This may take a moment (OCR active).")

                # Initialize splitter
                splitter = PDFSplitter(
                    ocr_engine=ocr_engine,
                    use_llm=use_llm,
                    use_llm_ocr=use_llm_ocr,
                    api_key=api_key if (use_llm or use_llm_ocr) else None,
                    api_base=api_base if (use_llm or use_llm_ocr) else None,
                    llm_model=llm_model,
                    azure_deployment=azure_deployment,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version
                )

                # Create output directory (persistent)
                output_dir = "split_forms"
                os.makedirs(output_dir, exist_ok=True)

                # Run the split logic
                documents = splitter.split_pdf(
                    input_path, 
                    output_dir, 
                    use_parallel=use_parallel, 
                    max_workers=max_workers
                )

                status_container.success(f"✅ Successfully split into {len(documents)} document(s)!")

                # Show warnings for low-confidence or ambiguous documents
                low_confidence_docs = splitter.get_low_confidence_documents(documents)
                ambiguous_docs = splitter.get_ambiguous_documents(documents)

                if low_confidence_docs or ambiguous_docs:
                    st.warning("⚠️ **Quality Warnings** - Some documents may need manual review:")
                    
                    warning_cols = st.columns(2)
                    
                    if low_confidence_docs:
                        with warning_cols[0]:
                            st.markdown(f"**🔍 Low Confidence ({len(low_confidence_docs)})**")
                            for doc in low_confidence_docs:
                                st.caption(f"• {doc['type']} #{doc['id']} (confidence: {doc.get('confidence', 'N/A')}%)")
                    
                    if ambiguous_docs:
                        with warning_cols[1]:
                            st.markdown(f"**⚠️ Ambiguous Pages ({len(ambiguous_docs)})**")
                            for doc in ambiguous_docs:
                                forms = doc.get('ambiguous_forms', [])
                                st.caption(f"• {doc['type']} #{doc['id']} (also contains: {', '.join(forms)})")
                    
                    st.divider()

                # Display results in a grid or list
                if documents:
                    st.markdown(f"**📋 Found {len(documents)} documents:**")

                    for i, doc in enumerate(documents):
                        # Generate the filename again to locate the file
                        filename = splitter.generate_filename(doc)
                        filepath = os.path.join(output_dir, filename)

                        # Create a card-like layout for each document
                        with st.container():
                            # Add visual indicator for warnings
                            if doc.get('confidence', 100) < 80 or doc.get('is_ambiguous', False):
                                st.markdown(f"<div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns([3, 1, 1])

                            with col1:
                                status_icons = []
                                if doc.get('confidence', 100) < 80:
                                    status_icons.append("🔍")
                                if doc.get('is_ambiguous', False):
                                    status_icons.append("⚠️")
                                if doc.get('method') == 'llm':
                                    status_icons.append("🤖")
                                
                                icon_str = " ".join(status_icons) if status_icons else "✅"
                                st.markdown(f"**{icon_str} {i+1}. {doc['type']}**")
                                
                                page_info = f"Pages: {len(doc['pages'])} | ID: {doc['id']}"
                                if show_debug:
                                    page_info += f" | Confidence: {doc.get('confidence', 'N/A')}%"
                                    if doc.get('method') == 'llm':
                                        page_info += " | 🤖 LLM"
                                st.caption(page_info)

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
                            
                            with col3:
                                if show_debug:
                                    patterns = doc.get('matched_patterns', [])
                                    if patterns:
                                        st.caption(f"**Patterns:** {', '.join(patterns)}")
                                    if doc.get('reasoning'):
                                        st.caption(f"**Reasoning:** {doc['reasoning'][:100]}...")
                            
                            if doc.get('confidence', 100) < 80 or doc.get('is_ambiguous', False):
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                            st.divider()

            except Exception as e:
                status_container.error(f"An error occurred during processing: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
