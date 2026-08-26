import argparse
import json
import os
 
# ------------------------------------------------------------
# Phải đặt TRƯỚC import paddle/paddleocr (qua OCRProcessor) để
# tránh lỗi "Descriptors cannot be created directly" khi protobuf
# trong venv mới hơn bản PaddlePaddle được biên dịch sẵn.
# ------------------------------------------------------------
 
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
 
from pathlib import Path
 
from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_processor import OCRProcessor
from src.inference.input_builder import LayoutLMInputBuilder
from src.inference.layoutlmv3_inference import LayoutLMv3Inference
from src.processing.invoice_normalizer import InvoiceNormalizer
from src.knowledge_base.chroma_store import ChromaInvoiceStore
from src.pipeline.invoice_qa import InvoiceQA
from src.inference.invoice_inference import print_result
 
 
DEFAULT_MODEL_DIR = Path("models/layoutlmv3/best_model")
DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_INFERENCE_DIR = Path("outputs/inference")
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full invoice QA pipeline end-to-end."
    )
    parser.add_argument("--input", required=True, help="Invoice PDF/PNG/JPG.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--inference-dir", default=str(DEFAULT_INFERENCE_DIR))
    return parser.parse_args()
 
def extract_and_index(
    input_path: Path,
    model_dir: Path,
    chroma_dir: Path,
    inference_dir: Path,
) -> list:
 
    pages = InputManager().load(input_path)
    pages = ImagePreprocessor().process(pages)
 
    ocr_result = OCRProcessor().recognize(pages)
    if not ocr_result.pages:
        raise SystemExit("[ERROR] OCR detected no pages.")
 
    box_builder = LayoutLMInputBuilder()
    model = LayoutLMv3Inference(model_path=str(model_dir))
    normalizer = InvoiceNormalizer()
 
    store = ChromaInvoiceStore(persist_directory=str(chroma_dir))
    inference_dir.mkdir(parents=True, exist_ok=True)
 
    base_name = input_path.stem
    is_multi_page = len(ocr_result.pages) > 1
 
    indexed_ids = []
 
    for image, page in zip(pages, ocr_result.pages):
 
        if not page.words:
            print(f"[WARNING] Page {page.page_number} has no OCR text. Skipping.")
            continue
 
        words = [w.text for w in page.words]
        boxes = [box_builder.quad_to_box(w.bbox) for w in page.words]
 
        structured = model.predict(image=image, words=words, boxes=boxes)
        normalized = normalizer.normalize(structured)
 
        invoice_id = (
            f"{base_name}_p{page.page_number}" if is_multi_page else base_name
        )
 
        print_result(f"NORMALIZED INVOICE — {invoice_id}", normalized)
 
        store.add_invoice(invoice_id, normalized)
 
        json_path = inference_dir / f"{invoice_id}_inference.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "input_file": str(input_path),
                    "page_number": page.page_number,
                    "structured": structured,
                    "normalized": normalized,
                },
                file,
                indent=4,
                ensure_ascii=False,
            )
 
        print(f"[OK] Indexed invoice '{invoice_id}' into {chroma_dir}")
        indexed_ids.append(invoice_id)
 
    if not indexed_ids:
        raise SystemExit("[ERROR] No page produced a valid invoice.")
 
    return indexed_ids
 
def main():
    args = parse_args()
 
    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    chroma_dir = Path(args.chroma_dir)
    inference_dir = Path(args.inference_dir)
 
    print()
    print("=" * 70)
    print("INVOICE PIPELINE — EXTRACT + INDEX")
    print("=" * 70)
    print(f"Input : {input_path}")
 
    indexed_ids = extract_and_index(input_path, model_dir, chroma_dir, inference_dir)
 
    print()
    print(f"[OK] Indexed {len(indexed_ids)} invoice(s): {', '.join(indexed_ids)}")
 
    print()
    print("=" * 70)
    print("QUESTION ANSWERING")
    print("=" * 70)
    print(f"Question: {args.question}")
 
    qa = InvoiceQA(top_k=5, min_relevance_score=0.80)
    result = qa.ask(args.question)
 
    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(f"Intent  : {result.get('intent', '')}")
    print(f"Answer  : {result.get('answer', '')}")
    print(f"Success : {result.get('success', False)}")
 
    data = result.get("data")
    if data:
        print(f"Data    : {data}")
 
 
if __name__ == "__main__":
    main()
 