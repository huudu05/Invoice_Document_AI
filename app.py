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
 
 
@st.cache_resource(show_spinner="Đang tải mô hình OCR + LayoutLMv3...")
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
        raise ValueError("Không phát hiện được văn bản trên hóa đơn này.")
 
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
        raise ValueError("Không phát hiện được văn bản trên bất kỳ trang nào.")
 
    return results
 
 
st.sidebar.title("⚙️ Cấu hình")
model_dir = st.sidebar.text_input("Model LayoutLMv3", value=DEFAULT_MODEL_DIR)
chroma_dir = st.sidebar.text_input("Thư mục ChromaDB", value=DEFAULT_CHROMA_DIR)
inference_dir = st.sidebar.text_input("Thư mục lưu JSON inference", value=DEFAULT_INFERENCE_DIR)
 
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Invoice Document AI**\n\n"
    "Trích xuất & tra cứu / phân tích hóa đơn "
    "bằng LayoutLMv3 + RAG + LLM."
)
 
st.title("📄 Invoice Document AI")
st.caption("Upload hóa đơn → trích xuất tự động → hỏi đáp & phân tích")
 
if "indexed_invoices" not in st.session_state:
    st.session_state.indexed_invoices = []
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
 
st.header("1. Tải hóa đơn lên")
 
uploaded_files = st.file_uploader(
    "Chọn ảnh hoặc PDF hóa đơn (PDF nhiều trang = nhiều hóa đơn, "
    "mỗi trang được xử lý riêng)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
)
 
if uploaded_files and st.button("🔍 Trích xuất & Lưu vào hệ thống", type="primary"):
    pipeline = load_extraction_pipeline(model_dir)
 
    progress = st.progress(0, text="Đang xử lý...")
 
    for index, uploaded_file in enumerate(uploaded_files):
        try:
            page_results = extract_and_index(
                uploaded_file, pipeline, chroma_dir, inference_dir
            )
            st.session_state.indexed_invoices.extend(page_results)
            st.success(
                f"✅ Đã trích xuất & lưu {len(page_results)} hóa đơn "
                f"từ: {uploaded_file.name}"
            )
        except Exception as error:
            st.error(f"❌ Lỗi khi xử lý {uploaded_file.name}: {error}")
 
        progress.progress((index + 1) / len(uploaded_files))
 
    progress.empty()
 
    st.session_state.pop("qa_engine", None)
 
 
if st.session_state.indexed_invoices:
    st.header("2. Hóa đơn đã trích xuất trong phiên này")
 
    for item in reversed(st.session_state.indexed_invoices):
        with st.expander(f"📄 {item['invoice_id']}"):
            col_image, col_data = st.columns([1, 1])
 
            with col_image:
                st.image(item["image"], caption="Ảnh OCR (box + id)", use_container_width=True)
 
            with col_data:
                st.markdown("**Thông tin chuẩn hóa:**")
                st.table({
                    "Trường": ["Công ty", "Ngày", "Địa chỉ", "Tổng tiền"],
                    "Giá trị": [
                        item["normalized"].get("company", ""),
                        item["normalized"].get("date", ""),
                        item["normalized"].get("address", ""),
                        item["normalized"].get("total", ""),
                    ],
                })
 
 
st.header("3. Hỏi đáp & Phân tích")
 
st.caption(
    "Ví dụ: 'Hóa đơn của X có tổng tiền bao nhiêu?', "
    "'Tổng chi tiêu tất cả hóa đơn là bao nhiêu?', "
    "'Công ty nào chi nhiều nhất?', "
    "'So sánh chi tiêu giữa A và B'."
)
 
question = st.text_input("Nhập câu hỏi", key="question_input")
 
if st.button("💬 Hỏi") and question.strip():
    qa, error = get_qa_engine(chroma_dir)
 
    if qa is None:
        st.warning(
            "Chưa có hóa đơn nào trong hệ thống. "
            "Vui lòng upload ít nhất 1 hóa đơn ở Mục 1 trước khi hỏi."
        )
    else:
        with st.spinner("Đang tìm câu trả lời..."):
            result = qa.ask(question)
 
        st.session_state.chat_history.append((question, result))
 
if st.session_state.chat_history:
    st.markdown("---")
    for query, result in reversed(st.session_state.chat_history):
        st.markdown(f"**🙋 Câu hỏi:** {query}")
 
        if result.get("success"):
            st.success(result.get("answer", ""))
        else:
            st.info(result.get("answer", ""))
 
        with st.expander("Chi tiết"):
            st.write(f"Intent: `{result.get('intent')}`")
            st.write(f"Nguồn câu trả lời: `{result.get('answer_source', '')}`")
 
            source = result.get("source")
            if source:
                metadata = source.get("metadata", {})
                st.write(f"Hóa đơn nguồn: **{metadata.get('company', '')}**")
                st.json(metadata)
 
            data = result.get("data")
            if data:
                st.write("Dữ liệu tính toán (analytics):")
                st.json(data)
 
        st.markdown("---")
 