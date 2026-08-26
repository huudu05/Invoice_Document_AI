import argparse
import torch
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
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full invoice QA pipeline end-to-end."
    )
    parser.add_argument("--input", required=True, help="Invoice PDF/PNG/JPG.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    return parser.parse_args()
 
 
def extract_and_index(input_path: Path, model_dir: Path, chroma_dir: Path) -> str:
    """Run OCR + LayoutLMv3 + normalize on one invoice, index it, return its id."""
 
    pages = InputManager().load(input_path)
    pages = ImagePreprocessor().process(pages)
 
    ocr_result = OCRProcessor().recognize(pages)
    if not ocr_result.pages or not ocr_result.pages[0].words:
        raise SystemExit("[ERROR] OCR detected no text.")
 
    page = ocr_result.pages[0]
    image = pages[0]
 
    box_builder = LayoutLMInputBuilder()
    words = [w.text for w in page.words]
    boxes = [box_builder.quad_to_box(w.bbox) for w in page.words]
 
    model = LayoutLMv3Inference(model_path=str(model_dir))
    structured = model.predict(image=image, words=words, boxes=boxes)
 
    normalized = InvoiceNormalizer().normalize(structured)
    print_result("NORMALIZED INVOICE", normalized)
 
    invoice_id = input_path.stem
    store = ChromaInvoiceStore(persist_directory=str(chroma_dir))
    store.add_invoice(invoice_id, normalized)
 
    print(f"[OK] Indexed invoice '{invoice_id}' into {chroma_dir}")
    return invoice_id
 
 
def main():
    args = parse_args()
 
    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    chroma_dir = Path(args.chroma_dir)
 
    extract_and_index(input_path, model_dir, chroma_dir)
 
    qa = InvoiceQA(top_k=5, min_relevance_score=0.80)
    result = qa.ask(args.question)
 
    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.get("answer", ""))
    print(f"\nSuccess: {result.get('success', False)}")
 
 
if __name__ == "__main__":
    main()