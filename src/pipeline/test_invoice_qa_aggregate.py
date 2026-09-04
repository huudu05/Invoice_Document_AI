from src.pipeline.invoice_qa import InvoiceQA
from src.analytics.invoice_analytics import InvoiceAnalytics
 
 
def print_case(title: str, query: str, result: dict) -> None:
    print()
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(f"Query   : {query}")
    print(f"Intent  : {result.get('intent')}")
    print(f"Success : {result.get('success')}")
    print(f"Source  : {result.get('answer_source')}")
    print(f"Answer  :\n{result.get('answer')}")
    if "data" in result:
        print(f"Data    : {result['data']}")
 
 
def main():
    qa = InvoiceQA(top_k=5, min_relevance_score=0.80)
 
    analytics = InvoiceAnalytics(store=qa.retriever.store)
 
    print("=" * 80)
    print("CURRENT DATA IN THE SYSTEM")
    print("=" * 80)
    print(f"Total number of invoices: {len(analytics.invoices)}")
    for company, amount in analytics.spending_by_company().items():
        print(f"  {company:<35} total: {amount}")
 
    if len(analytics.invoices) < 2:
        print()
        print("[WARNING] At least 2 invoices from DIFFERENT COMPANIES are required "
              "for this test to be meaningful (currently not enough). "
              "Please run invoice_inference.py on a few different images first.")
        return
 
    companies = list(analytics.spending_by_company().keys())
    company_a = companies[0]
    company_b = companies[1] if len(companies) > 1 else companies[0]
 


    # TEST 1 — aggregate_sum (tổng tất cả)
    expected_total = analytics.total_spending()
    result = qa.ask("What is the total spending across all invoices?")
    print_case("TEST 1: aggregate_sum (all invoices)", "What is the total spending across all invoices?", result)
    assert result["success"] is True, "TEST 1 FAILED: success must be True"
    assert result["intent"] == "aggregate_sum", f"TEST 1 FAILED: incorrect intent ({result['intent']})"
    assert result["data"]["total"] == expected_total, (
        f"TEST 1 FAILED: total mismatch —  system returned {result['data']['total']}, "
        f"expected {expected_total}"
    )
    print("[PASS] TEST 1")
 

    # TEST 2 — aggregate_sum theo công ty
    expected_company_total = analytics.total_spending(company=company_a)
    result = qa.ask(f"What is the total spending for {company_a}?")
    print_case("TEST 2: aggregate_sum (by company)", f"What is the total spending for {company_a}?", result)
    assert result["success"] is True
    assert result["data"]["total"] == expected_company_total, (
        f"TEST 2 FAILED: company total mismatch — "
        f"system returned {result['data']['total']}, "
        f"expected {expected_company_total}"
    )
    print("[PASS] TEST 2")
 


    # TEST 3 — aggregate_count
    expected_count = analytics.count_invoices()
    result = qa.ask("How many invoices are in the system?")
    print_case("TEST 3: aggregate_count", "How many invoices are in the system?", result)
    assert result["intent"] == "aggregate_count"
    assert result["data"]["count"] == expected_count, (
        f"TEST 3 FAILED: count mismatch — "
        f"system returned {result['data']['count']}, "
        f"expected {expected_count}"
    )
    print("[PASS] TEST 3")
 

    # TEST 4 — aggregate_top
    expected_top = analytics.top_companies(n=5)
    result = qa.ask("Which company has the highest total spending?")
    print_case("TEST 4: aggregate_top", "Which company has the highest total spending?", result)
    assert result["intent"] == "aggregate_top"
    assert result["data"]["ranking"] == expected_top, "TEST 4 FAILED: ranking mismatch"
    print("[PASS] TEST 4")
 


    # TEST 5 — aggregate_compare
    result = qa.ask(f"Compare spending between {company_a} and {company_b}")
    print_case("TEST 5: aggregate_compare", f"Compare spending between {company_a} and {company_b}", result)
    assert result["intent"] == "aggregate_compare"
    print("[PASS] TEST 5 (please review the answer above)")

 
    # TEST 6 — Regression
    result = qa.ask(f"What is the total amount of the {company_a} invoice?")
    print_case("TEST 6: regression (single-invoice, not aggregate)",
               f"What is the total amount of the {company_a} invoice?", result)
    assert result["intent"] == "total", (
        f"TEST 6 FAILED: a specific invoice question was incorrectly "
        f"'{result['intent']}' instead of 'total'"
    )
    print("[PASS] TEST 6")
 
    print()
    print("=" * 80)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 80)
 
 
if __name__ == "__main__":
    main()
 