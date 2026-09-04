import json
import os
import tempfile
from pathlib import Path
import streamlit as st
 
from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_processor import OCRProcessor
from src.ocr.visualizer import OCRVisualizer
from src.inference.input_builder import LayoutLMInputBuilder
from src.inference.layoutlmv3_inference import LayoutLMv3Inference
from src.processing.invoice_normalizer import InvoiceNormalizer
from src.knowledge_base.chroma_store import ChromaInvoiceStore
from src.pipeline.invoice_qa import InvoiceQA
 
 
DEFAULT_MODEL_DIR = "models/layoutlmv3/best_model"
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_INFERENCE_DIR = "outputs/inference"
 
st.set_page_config(page_title="Invoice Document AI", layout="wide")
 
 
@st.cache_resource(show_spinner="Loading OCR + LayoutLMv3 models...")
def load_extraction_pipeline(model_dir: str):
    return {
        "input_manager": InputManager(),
        "preprocessor": ImagePreprocessor(),
        "ocr": OCRProcessor(),
        "model": LayoutLMv3Inference(model_path=model_dir),
        "normalizer": InvoiceNormalizer(),
        "box_builder": LayoutLMInputBuilder(),
        "visualizer": OCRVisualizer(),
    }
 
 
def get_qa_engine(chroma_dir: str):
    if "qa_engine" not in st.session_state:
        try:
            st.session_state.qa_engine = InvoiceQA(
                top_k=5, min_relevance_score=0.80
            )
        except RuntimeError as error:
            st.session_state.qa_engine = None
            return None, str(error)
    return st.session_state.qa_engine, None
 
 
def extract_and_index(
    uploaded_file,
    pipeline: dict,
    chroma_dir: str,
    inference_dir: str,
) -> list:
 
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp_file.name)
 
    pages = pipeline["input_manager"].load(tmp_path)
    pages = pipeline["preprocessor"].process(pages)
 
    ocr_result = pipeline["ocr"].recognize(pages)
    if not ocr_result.pages:
        raise ValueError("No text was detected on this invoice.")
 
    base_name = Path(uploaded_file.name).stem
    is_multi_page = len(ocr_result.pages) > 1
 
    store = ChromaInvoiceStore(persist_directory=chroma_dir)
    Path(inference_dir).mkdir(parents=True, exist_ok=True)
 
    results = []
 
    for image, page in zip(pages, ocr_result.pages):
 
        if not page.words:
            continue
 
        words = [w.text for w in page.words]
        boxes = [pipeline["box_builder"].quad_to_box(w.bbox) for w in page.words]
 
        structured = pipeline["model"].predict(image=image, words=words, boxes=boxes)
        normalized = pipeline["normalizer"].normalize(structured)
 
        invoice_id = (
            f"{base_name}_p{page.page_number}" if is_multi_page else base_name
        )
 
        store.add_invoice(invoice_id, normalized)
 
        json_path = Path(inference_dir) / f"{invoice_id}_inference.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "input_file": uploaded_file.name,
                    "page_number": page.page_number,
                    "structured": structured,
                    "normalized": normalized,
                },
                file,
                indent=4,
                ensure_ascii=False,
            )
 
        vis_image = pipeline["visualizer"].visualize_page(image, page)
 
        results.append({
            "invoice_id": invoice_id,
            "structured": structured,
            "normalized": normalized,
            "image": vis_image,
        })
 
    if not results:
        raise ValueError("No text was detected on any invoice page.")
 
    return results
 
 
st.sidebar.title("⚙️ Configuration")
model_dir = st.sidebar.text_input("Model LayoutLMv3", value=DEFAULT_MODEL_DIR)
chroma_dir = st.sidebar.text_input("ChromaDB directory", value=DEFAULT_CHROMA_DIR)
inference_dir = st.sidebar.text_input("Inference JSON output directory", value=DEFAULT_INFERENCE_DIR)
 
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Invoice Document AI**\n\n"
    "Invoice extraction, search, and analytics "
    "using LayoutLMv3 + RAG + LLM."
)
 
st.title("📄 Invoice Document AI")
st.caption("Upload invoice → automatic extraction → question answering & analysis")
 
if "indexed_invoices" not in st.session_state:
    st.session_state.indexed_invoices = []
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
 
st.header("1. Upload invoices")
 
uploaded_files = st.file_uploader(
    "Select invoice images or PDF files "
    "(multi-page PDFs are processed as separate invoices)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
)
 
if uploaded_files and st.button("🔍 Extract and Save", type="primary"):
    pipeline = load_extraction_pipeline(model_dir)
 
    progress = st.progress(0, text="Processing...")
 
    for index, uploaded_file in enumerate(uploaded_files):
        try:
            page_results = extract_and_index(
                uploaded_file, pipeline, chroma_dir, inference_dir
            )
            st.session_state.indexed_invoices.extend(page_results)
            st.success(
                f"✅ Extracted and saved {len(page_results)} invoices "
                f"from: {uploaded_file.name}"
            )
        except Exception as error:
            st.error(f"❌ Error processing {uploaded_file.name}: {error}")
 
        progress.progress((index + 1) / len(uploaded_files))
 
    progress.empty()
 
    st.session_state.pop("qa_engine", None)
 
 
if st.session_state.indexed_invoices:
    st.header("2. Invoices extracted in this session")
 
    for item in reversed(st.session_state.indexed_invoices):
        with st.expander(f"📄 {item['invoice_id']}"):
            col_image, col_data = st.columns([1, 1])
 
            with col_image:
                st.image(item["image"], caption="OCR Image (boxes + IDs)", use_container_width=True)
 
            with col_data:
                st.markdown("**Normalized Information:**")
                st.table({
                    "Field": ["Company", "Date", "Address", "Total"],
                    "Value": [
                        item["normalized"].get("company", ""),
                        item["normalized"].get("date", ""),
                        item["normalized"].get("address", ""),
                        item["normalized"].get("total", ""),
                    ],
                })
 
 
st.header("3. Questions and Analytics")
 
st.caption(
    "Examples: 'What is the total amount of the invoice from X?', "
    "'What is the total spending across all invoices?', "
    "'Which company has the highest spending?', "
    "'Compare spending between A and B'."
)
 
question = st.text_input("Enter your question", key="question_input")
 
if st.button("💬 Ask") and question.strip():
    qa, error = get_qa_engine(chroma_dir)
 
    if qa is None:
        st.warning(
            "There are no invoices in the system. "
            "Please upload at least one invoice in Section 1 before asking a question."
        )
    else:
        with st.spinner("Finding an answer..."):
            result = qa.ask(question)
 
        st.session_state.chat_history.append((question, result))
 
if st.session_state.chat_history:
    st.markdown("---")
    for query, result in reversed(st.session_state.chat_history):
        st.markdown(f"**🙋 Question:** {query}")
 
        if result.get("success"):
            st.success(result.get("answer", ""))
        else:
            st.info(result.get("answer", ""))
 
        with st.expander("Details"):
            st.write(f"Intent: `{result.get('intent')}`")
            st.write(f"Answer Source: `{result.get('answer_source', '')}`")
 
            source = result.get("source")
            if source:
                metadata = source.get("metadata", {})
                st.write(f"Source Invoice: **{metadata.get('company', '')}**")
                st.json(metadata)
 
            data = result.get("data")
            if data:
                st.write("Computed Data (analytics):")
                st.json(data)
 
        st.markdown("---")
 