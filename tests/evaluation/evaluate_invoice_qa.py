import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
 
from src.pipeline.invoice_qa import InvoiceQA
from src.analytics.invoice_analytics import InvoiceAnalytics
 
 
def load_golden_set(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
 

def check_field_value(case: Dict[str, Any], result: Dict[str, Any]) -> bool:
    expected = str(case.get("expected_value", "")).strip().lower()
    answer = str(result.get("answer", "")).strip().lower()
    if not expected:
        return False
    return expected in answer
 
 
def check_refusal(result: Dict[str, Any]) -> bool:
    if result.get("success") is False:
        return True
    answer = str(result.get("answer", "")).lower()
    refusal_markers = ["not found", "no matching", "no relevant", "please enter a question"]
    return any(marker in answer for marker in refusal_markers)
 
 
def compute_expected_number(case: Dict[str, Any], analytics: InvoiceAnalytics):

    intent = case["intent"]
    company = case.get("company")
 
    if intent == "aggregate_sum":
        return analytics.total_spending(company=company)
    if intent == "aggregate_count":
        return analytics.count_invoices(company=company)
    if intent == "aggregate_top":
        return analytics.top_companies(n=5)
    return None
 
 
def check_exact_number(
    case: Dict[str, Any],
    result: Dict[str, Any],
    analytics: InvoiceAnalytics,
    tolerance: float = 0.01,
) -> Optional[bool]:
    expected = compute_expected_number(case, analytics)
    if expected is None:
        return None
 
    data = result.get("data") or {}
    intent = case["intent"]
 
    if intent == "aggregate_sum":
        actual = data.get("total")
    elif intent == "aggregate_count":
        actual = data.get("count")
    elif intent == "aggregate_top":
        actual = data.get("ranking")
    else:
        actual = None
 
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= tolerance
    return expected == actual
 
 
# RUN + REPORT
 
def evaluate_case(case: Dict[str, Any], qa: InvoiceQA, analytics: InvoiceAnalytics) -> Dict[str, Any]:
    question = case.get("question", "")
 
    if not question.strip():
        result: Dict[str, Any] = {"intent": "general", "answer": "", "success": False}
    else:
        result = qa.ask(question)
 
    expected_intent = case.get("intent", "")
    predicted_intent = result.get("intent", "")
    intent_correct = predicted_intent == expected_intent
 
    expected_type = case.get("expected_type")
 
    if expected_type == "field_value":
        answer_correct = check_field_value(case, result)
    elif expected_type == "refusal":
        answer_correct = check_refusal(result)
    elif expected_type == "exact_number":
        answer_correct = check_exact_number(case, result, analytics)
    else:
        answer_correct = None  # manual_review or unknown type
 
    return {
        "id": case.get("id", ""),
        "question": question,
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
        "intent_correct": intent_correct,
        "expected_type": expected_type,
        "answer_correct": answer_correct,
        "answer": result.get("answer", ""),
        "success": result.get("success"),
    }
 
 
def run_evaluation(
    golden_set: List[Dict[str, Any]],
    qa: InvoiceQA,
    analytics: InvoiceAnalytics,
) -> List[Dict[str, Any]]:
    return [evaluate_case(case, qa, analytics) for case in golden_set]
 
 
def summarize(rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    intent_correct = sum(1 for row in rows if row["intent_correct"])
 
    checkable = [row for row in rows if row["answer_correct"] is not None]
    answer_correct = sum(1 for row in checkable if row["answer_correct"])
 
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions           : {total}")
    print(f"Intent accuracy           : {intent_correct}/{total} "
          f"({intent_correct / total:.1%})" if total else "N/A")
    if checkable:
        print(f"Answer accuracy (checkable): {answer_correct}/{len(checkable)} "
              f"({answer_correct / len(checkable):.1%})")
    print(f"Needs manual review        : {total - len(checkable)}")
 
    print()
    print("Breakdown by expected intent:")
    intents = sorted({row["expected_intent"] for row in rows})
    for intent in intents:
        subset = [row for row in rows if row["expected_intent"] == intent]
        correct = sum(1 for row in subset if row["intent_correct"])
        print(f"  {intent:<20} {correct}/{len(subset)}")
 
    failures = [
        row for row in rows
        if not row["intent_correct"] or row["answer_correct"] is False
    ]
    if failures:
        print()
        print(f"Failed cases ({len(failures)}):")
        for row in failures:
            print(f"  [{row['id']}] {row['question']}")
            print(f"      expected_intent={row['expected_intent']} "
                  f"predicted_intent={row['predicted_intent']}")
            print(f"      answer: {row['answer'][:120]}")
 
 
def save_report(rows: List[Dict[str, Any]], report_path: Path) -> None:
    if not rows:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate InvoiceQA end-to-end accuracy against a golden test set."
    )
    parser.add_argument(
        "--golden-set", default="tests/evaluation/qa_golden_set.json",
        help="Path to the golden set JSON file.",
    )
    parser.add_argument(
        "--report", default="outputs/evaluation/qa_report.csv",
        help="Path to write the detailed CSV report.",
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    qa = InvoiceQA(top_k=5, min_relevance_score=0.80)
    analytics = InvoiceAnalytics(store=qa.retriever.store)
 
    golden_set = load_golden_set(Path(args.golden_set))
    rows = run_evaluation(golden_set, qa, analytics)
 
    summarize(rows)
    save_report(rows, Path(args.report))
 
    print()
    print(f"[OK] Detailed report saved to: {args.report}")
 
 
if __name__ == "__main__":
    main()
 