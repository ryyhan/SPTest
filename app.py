import streamlit as st
import os
import tempfile
from DocSplitter import PDFSplitter
from llm_config import create_azure_config, LLMConfig

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
            "LLM Vision (GPT-4o - Most accurate, requires Azure OpenAI)"
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
        
        # Show file info
        st.subheader("📋 File Ready")
        st.info(f"**File:** {uploaded_file.name} ({round(uploaded_file.size / 1024, 2)} KB)")
        
        # Processing button - only show after file is uploaded
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
        
        if process_button:
            # Validate LLM configuration if LLM features are enabled
            if use_llm or use_llm_ocr:
                # Try to load LLM config from llm_config.py
                try:
                    from llm_config import default_config
                    if not default_config.api_key or not default_config.azure_endpoint:
                        st.error("⚠️ **LLM configuration missing!** Please configure Azure OpenAI settings in `llm_config.py`")
                        st.code("""
# In llm_config.py, update the default_config:
default_config = create_azure_config(
    api_key="your-api-key-here",
    azure_deployment="your-deployment-name",
    azure_endpoint="https://your-resource.openai.azure.com/"
)
                        """)
                        st.stop()
                except Exception as e:
                    st.error(f"Error loading LLM configuration: {e}")
                    st.stop()
            
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

                    # Load LLM config from llm_config.py
                    llm_config = None
                    if use_llm or use_llm_ocr:
                        from llm_config import default_config
                        llm_config = default_config

                    # Initialize splitter
                    splitter = PDFSplitter(
                        ocr_engine=ocr_engine,
                        use_llm=use_llm,
                        use_llm_ocr=use_llm_ocr,
                        api_key=llm_config.api_key if llm_config else None,
                        llm_model=llm_config.classification_model if llm_config else "gpt-4o-mini",
                        azure_deployment=llm_config.azure_deployment if llm_config else None,
                        azure_endpoint=llm_config.azure_endpoint if llm_config else None,
                        azure_api_version=llm_config.azure_api_version if llm_config else None,
                        llm_config=llm_config
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
