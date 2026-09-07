import argparse
import json
import torch
from pathlib import Path
from typing import Any, Dict

from src.logging_config import get_logger
from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_processor import OCRProcessor
from src.inference.input_builder import LayoutLMInputBuilder
from src.inference.layoutlmv3_inference import LayoutLMv3Inference
from src.processing.invoice_normalizer import InvoiceNormalizer
 
DEFAULT_MODEL_DIR = Path("models/layoutlmv3/best_model")
DEFAULT_OUTPUT_DIR = Path("outputs/inference")

logger = get_logger(__name__)
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR + fine-tuned LayoutLMv3 on a real invoice."
    )
 
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to invoice file (PDF/PNG/JPG/JPEG).",
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
        help="Path to fine-tuned LayoutLMv3 model.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for inference outputs (JSON).",
    )
 
    return parser.parse_args()

 
def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
 
 
def print_result(title: str, fields: Dict[str, Any]) -> None:
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)
    for key, value in fields.items():
        logger.info("%-10s: %s", key, value if value else "[EMPTY]")
    logger.info("=" * 70)
 
 
def main():
    args = parse_args()
 
    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info("=" * 70)
    logger.info("INVOICE DOCUMENT AI")
    logger.info("Raw -> Structured -> Normalized")
    logger.info("=" * 70)
    logger.info("Input : %s", input_path)
    logger.info("Model : %s", model_dir)
    logger.info("=" * 70)

 
    # 1) Load (PDF or image) via the shared InputManager.
    try:
        pages = InputManager().load(input_path)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(f"[ERROR] {error}") from error
 
    if not pages:
        raise SystemExit("[ERROR] No pages were loaded from the input file.")

 
    # 2) Preprocess (RGB convert + resize) via shared preprocessor.
    pages = ImagePreprocessor().process(pages)

 
    # 3) OCR via shared OCRProcessor (PaddleOCR wrapper).
    try:
        ocr_result = OCRProcessor().recognize(pages)
    except Exception as error:
        logger.exception("OCR failed")
        raise SystemExit(f"[ERROR] OCR failed: {error}") from error
 
    if not ocr_result.pages or not ocr_result.pages[0].words:
        raise SystemExit("[ERROR] OCR detected no text on the input document.")
 
    page = ocr_result.pages[0]

    box_builder = LayoutLMInputBuilder()
    layout_input = box_builder.build(pages, ocr_result)[0]

    words = layout_input.words
    boxes = layout_input.boxes

    ocr_scores = [word.confidence for word in page.words]
    
 
    logger.info("[OK] OCR detected %d text boxes.", len(words))
 
    # 4) Fine-tuned LayoutLMv3 inference.
    try:
        model = LayoutLMv3Inference(model_path=str(model_dir))
        structured = model.predict(image=layout_input.image, words=words, boxes=boxes)
    except Exception as error:
        logger.exception("LayoutLMv3 inference failed")
        raise SystemExit(f"[ERROR] LayoutLMv3 inference failed: {error}") from error
 
    print_result("EXTRACTED INVOICE INFORMATION (RAW)", structured)

 
    # 5) Normalize
    normalized = InvoiceNormalizer().normalize(structured)
    print_result("NORMALIZED INVOICE", normalized)
 

    # 6) Persist result
    result = {
        "input_file": str(input_path),
        "model": str(model_dir),
        "structured": structured,
        "normalized": normalized,
        "ocr": {
            "num_words": len(words),
            "words": words,
            "boxes": boxes,
            "confidence": ocr_scores,
        },
    }
 
    output_path = output_dir / f"{input_path.stem}_inference.json"
    save_json(result, output_path)
 
    logger.info("[OK] Result saved to: %s", output_path)
    logger.info("INFERENCE COMPLETED.")
 
 
if __name__ == "__main__":
    main()
 