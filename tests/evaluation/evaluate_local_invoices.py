import argparse
import json
from pathlib import Path
from typing import Any, Dict
 
from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_processor import OCRProcessor
from src.inference.input_builder import LayoutLMInputBuilder
from src.inference.layoutlmv3_inference import LayoutLMv3Inference
from src.processing.invoice_normalizer import InvoiceNormalizer
 
from tests.evaluation.pipeline_eval_utils import FIELDS, compare_fields, run_pipeline_on_image
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the full pipeline on your own photographed invoices."
    )
    parser.add_argument("--images-dir", required=True, help="Folder containing the invoice images.")
    parser.add_argument("--ground-truth", required=True, help="Path to the hand-labeled ground truth JSON.")
    parser.add_argument("--model-dir", default="models/layoutlmv3/best_model")
    parser.add_argument("--report", default="outputs/evaluation/local_invoices_report.json")
    args = parser.parse_args()
 
    images_dir = Path(args.images_dir)
    with open(args.ground_truth, "r", encoding="utf-8") as file:
        ground_truth_map: Dict[str, Dict[str, Any]] = json.load(file)
 
    if not ground_truth_map:
        raise SystemExit(f"[ERROR] Ground truth file {args.ground_truth} is empty.")
 
    input_manager = InputManager()
    preprocessor = ImagePreprocessor()
    ocr = OCRProcessor()
    box_builder = LayoutLMInputBuilder()
    model = LayoutLMv3Inference(model_path=args.model_dir)
    normalizer = InvoiceNormalizer()
 
    per_field_correct = {name: 0 for name in FIELDS}
    per_field_total = {name: 0 for name in FIELDS}
    all_fields_correct_count = 0
    failures = 0
    rows = []
 
    for filename, ground_truth_raw in ground_truth_map.items():
        file_path = images_dir / filename
 
        if not file_path.exists():
            print(f"[WARNING] {filename} is listed in ground truth but not found in {images_dir}. Skipping.")
            failures += 1
            rows.append({"file": filename, "error": "file not found"})
            continue

        try:
            pages = input_manager.load(file_path)
        except (FileNotFoundError, ValueError) as error:
            print(f"[WARNING] Could not load {filename}: {error}. Skipping.")
            failures += 1
            rows.append({"file": filename, "error": str(error)})
            continue
 
        ground_truth_normalized = normalizer.normalize(ground_truth_raw)
 
        predicted, raw_structured, error = run_pipeline_on_image(
            pages[0], preprocessor, ocr, box_builder, model, normalizer
        )
 
        if predicted is None:
            failures += 1
            rows.append({"file": filename, "error": error})
            print(f"[WARNING] {filename}: {error}")
            continue
 
        field_results = compare_fields(predicted, ground_truth_normalized)
 
        for field_name in FIELDS:
            per_field_total[field_name] += 1
            if field_results[field_name]:
                per_field_correct[field_name] += 1
 
        if all(field_results.values()):
            all_fields_correct_count += 1
 
        rows.append({
            "file": filename,
            "ground_truth": ground_truth_normalized,
            "raw_structured": raw_structured,
            "predicted": predicted,
            "field_results": field_results,
        })
 
        print(f"[OK] {filename}: {field_results}")
 
    n_evaluated = len(ground_truth_map) - failures
 
    print()
    print("=" * 70)
    print("REAL-WORLD (SELF-COLLECTED) PIPELINE EVALUATION")
    print("=" * 70)
    print(f"Images in ground truth: {len(ground_truth_map)}")
    print(f"Load/OCR failures      : {failures}")
    print(f"Images evaluated       : {n_evaluated}")
    print()
 
    for field_name in FIELDS:
        total = per_field_total[field_name]
        correct = per_field_correct[field_name]
        accuracy = correct / total if total else 0.0
        print(f"{field_name:<10}: {correct}/{total} ({accuracy:.1%})")
 
    if n_evaluated:
        print()
        print(
            f"All-fields-correct invoices: "
            f"{all_fields_correct_count}/{n_evaluated} "
            f"({all_fields_correct_count / n_evaluated:.1%})"
        )
 
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
 
    print()
    print(f"[OK] Detailed report saved to: {report_path}")
 
 
if __name__ == "__main__":
    main()
 