import difflib
 
from typing import Any, Dict, Optional, Tuple
 
FIELDS = ("company", "date", "address", "total")
 

COMPANY_SIMILARITY_THRESHOLD = 0.80
ADDRESS_SIMILARITY_THRESHOLD = 0.65
 
 
def normalize_for_comparison(value: str) -> str:
    value = str(value).lower()
    value = "".join(character if character.isalnum() or character.isspace() else " " for character in value)
    return " ".join(value.split())
 
 
def similarity_ratio(a: str, b: str) -> float:
    """difflib.SequenceMatcher ratio, 0.0 (nothing alike) to 1.0 (identical)."""
    return difflib.SequenceMatcher(None, a, b).ratio()
 
 
def compare_fields(
    predicted: Dict[str, Any],
    ground_truth_normalized: Dict[str, Any],
    company_threshold: float = COMPANY_SIMILARITY_THRESHOLD,
    address_threshold: float = ADDRESS_SIMILARITY_THRESHOLD,
) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
 
    thresholds = {"company": company_threshold, "address": address_threshold}
    for field_name, threshold in thresholds.items():
        expected = normalize_for_comparison(ground_truth_normalized.get(field_name, ""))
        actual = normalize_for_comparison(predicted.get(field_name, ""))
 
        if not expected:
            results[field_name] = True  # nothing to check against, skip
            continue
        if not actual:
            results[field_name] = False  # empty prediction is never a match
            continue
 
        results[field_name] = similarity_ratio(expected, actual) >= threshold
 
    results["date"] = predicted.get("date") == ground_truth_normalized.get("date")
 
    predicted_total = predicted.get("total")
    expected_total = ground_truth_normalized.get("total")
    if predicted_total is None or expected_total is None:
        results["total"] = predicted_total == expected_total
    else:
        results["total"] = abs(float(predicted_total) - float(expected_total)) <= 0.01
 
    return results
 
 
def run_pipeline_on_image(
    image,
    preprocessor,
    ocr,
    box_builder,
    model,
    normalizer,
    enhance: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    processed = preprocessor.process([image], enhance=enhance)[0]
 
    ocr_result = ocr.recognize([processed])
    if not ocr_result.pages or not ocr_result.pages[0].words:
        return None, None, "OCR detected no text"
 
    layout_inputs = box_builder.build([processed], ocr_result)
    layout_input = layout_inputs[0]
 
    structured = model.predict(
        image=layout_input.image,
        words=layout_input.words,
        boxes=layout_input.boxes,
    )
    return normalizer.normalize(structured), structured, None
 