import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
 
from datasets import load_dataset
 
from src.input_processing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_processor import OCRProcessor
from src.inference.input_builder import LayoutLMInputBuilder
from src.inference.layoutlmv3_inference import LayoutLMv3Inference
from src.processing.invoice_normalizer import InvoiceNormalizer
 
from tests.evaluation.pipeline_eval_utils import FIELDS, compare_fields, run_pipeline_on_image
 
 
FIELD_LABEL_MAP = {
    "s-company": "company",
    "s-date": "date",
    "s-address": "address",
    "s-total": "total",
}
 
 
def reconstruct_ground_truth_raw(
    words: List[str],
    ner_tags: List[int],
    id2label: Dict[int, str],
) -> Dict[str, str]:
    fields: Dict[str, List[str]] = {name: [] for name in FIELDS}
 
    for word, tag_id in zip(words, ner_tags):
        label = id2label.get(tag_id, "O").lower()
        field_name = FIELD_LABEL_MAP.get(label)
        if field_name:
            fields[field_name].append(word)
 
    return {name: " ".join(value).strip() for name, value in fields.items()}
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the full OCR + LayoutLMv3 pipeline on SROIE test images."
    )
    parser.add_argument("--model-dir", default="models/layoutlmv3/best_model")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N samples (omit to run the full test split).",
    )
    parser.add_argument("--report", default="outputs/evaluation/full_pipeline_report.json")
    parser.add_argument(
        "--enhance", action="store_true",
        help="Apply deskew/denoise/contrast enhancement before OCR (see "
             "ImagePreprocessor.process). Omit for the baseline run; run "
             "once with and once without this flag on the same --report "
             "path (renamed between runs) to compare accuracy.",
    )
    args = parser.parse_args()
 
    dataset = load_dataset("mp-02/sroie", split="test")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
 
    ner_feature = dataset.features["ner_tags"].feature
    id2label = {idx: label for idx, label in enumerate(ner_feature.names)}
 
    preprocessor = ImagePreprocessor()
    ocr = OCRProcessor()
    box_builder = LayoutLMInputBuilder()
    model = LayoutLMv3Inference(model_path=args.model_dir)
    normalizer = InvoiceNormalizer()
 
    per_field_correct = {name: 0 for name in FIELDS}
    per_field_total = {name: 0 for name in FIELDS}
    all_fields_correct_count = 0
    ocr_failures = 0
    rows = []
 
    for index, sample in enumerate(dataset):
        image = sample["image"].convert("RGB")
 
        ground_truth_raw = reconstruct_ground_truth_raw(
            sample["words"], sample["ner_tags"], id2label
        )
        # Run ground truth through the SAME normalizer as the prediction,
        # so e.g. "22/12/2017" (raw) vs "2017-12-22" (predicted) compare fairly.
        ground_truth_normalized = normalizer.normalize(ground_truth_raw)
 
        predicted, raw_structured, error = run_pipeline_on_image(
            image, preprocessor, ocr, box_builder, model, normalizer,
            enhance=args.enhance,
        )
 
        if predicted is None:
            ocr_failures += 1
            rows.append({"index": index, "error": error})
            continue
 
        field_results = compare_fields(predicted, ground_truth_normalized)
 
        for field_name in FIELDS:
            per_field_total[field_name] += 1
            if field_results[field_name]:
                per_field_correct[field_name] += 1
 
        if all(field_results.values()):
            all_fields_correct_count += 1
 
        rows.append({
            "index": index,
            "ground_truth": ground_truth_normalized,
            "raw_structured": raw_structured,
            "predicted": predicted,
            "field_results": field_results,
        })
 
        if (index + 1) % 20 == 0:
            print(f"[..] Processed {index + 1}/{len(dataset)} samples")
 
    n_evaluated = len(dataset) - ocr_failures
 
    print()
    print("=" * 70)
    print("FULL PIPELINE EVALUATION (own OCR + LayoutLMv3, SROIE test images)")
    print("=" * 70)
    print(f"Samples in test split : {len(dataset)}")
    print(f"OCR failures (skipped): {ocr_failures}")
    print(f"Samples evaluated     : {n_evaluated}")
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