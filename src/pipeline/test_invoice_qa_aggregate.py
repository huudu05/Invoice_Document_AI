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
    print("DỮ LIỆU HIỆN CÓ TRONG HỆ THỐNG")
    print("=" * 80)
    print(f"Tổng số hóa đơn: {len(analytics.invoices)}")
    for company, amount in analytics.spending_by_company().items():
        print(f"  {company:<35} tổng: {amount}")
 
    if len(analytics.invoices) < 2:
        print()
        print("[CẢNH BÁO] Cần index ít nhất 2 hóa đơn từ CÔNG TY KHÁC NHAU "
              "để test này có ý nghĩa (hiện tại chưa đủ). "
              "Hãy chạy invoice_inference.py trên vài ảnh khác nhau trước.")
        return
 
    companies = list(analytics.spending_by_company().keys())
    company_a = companies[0]
    company_b = companies[1] if len(companies) > 1 else companies[0]
 


    # TEST 1 — aggregate_sum (tổng tất cả)
    expected_total = analytics.total_spending()
    result = qa.ask("Tổng chi tiêu tất cả hóa đơn là bao nhiêu?")
    print_case("TEST 1: aggregate_sum (toàn bộ)", "Tổng chi tiêu tất cả hóa đơn là bao nhiêu?", result)
    assert result["success"] is True, "TEST 1 FAILED: success phải là True"
    assert result["intent"] == "aggregate_sum", f"TEST 1 FAILED: intent sai ({result['intent']})"
    assert result["data"]["total"] == expected_total, (
        f"TEST 1 FAILED: total lệch — hệ thống trả {result['data']['total']}, "
        f"tính tay ra {expected_total}"
    )
    print("[PASS] TEST 1")
 

    # TEST 2 — aggregate_sum theo công ty
    expected_company_total = analytics.total_spending(company=company_a)
    result = qa.ask(f"Tổng chi tiêu của {company_a} là bao nhiêu?")
    print_case("TEST 2: aggregate_sum (theo công ty)", f"Tổng chi tiêu của {company_a}...", result)
    assert result["success"] is True
    assert result["data"]["total"] == expected_company_total, (
        f"TEST 2 FAILED: total theo công ty lệch — "
        f"hệ thống {result['data']['total']}, tính tay {expected_company_total}"
    )
    print("[PASS] TEST 2")
 


    # TEST 3 — aggregate_count
    expected_count = analytics.count_invoices()
    result = qa.ask("Có bao nhiêu hóa đơn trong hệ thống?")
    print_case("TEST 3: aggregate_count", "Có bao nhiêu hóa đơn trong hệ thống?", result)
    assert result["intent"] == "aggregate_count"
    assert result["data"]["count"] == expected_count, (
        f"TEST 3 FAILED: count lệch — hệ thống {result['data']['count']}, "
        f"tính tay {expected_count}"
    )
    print("[PASS] TEST 3")
 

    # TEST 4 — aggregate_top
    expected_top = analytics.top_companies(n=5)
    result = qa.ask("Công ty nào có tổng chi tiêu cao nhất?")
    print_case("TEST 4: aggregate_top", "Công ty nào có tổng chi tiêu cao nhất?", result)
    assert result["intent"] == "aggregate_top"
    assert result["data"]["ranking"] == expected_top, "TEST 4 FAILED: ranking lệch"
    print("[PASS] TEST 4")
 


    # TEST 5 — aggregate_compare
    result = qa.ask(f"So sánh chi tiêu giữa {company_a} và {company_b}")
    print_case("TEST 5: aggregate_compare", f"So sánh {company_a} và {company_b}", result)
    assert result["intent"] == "aggregate_compare"
    print("[PASS] TEST 5 (kiểm tra bằng mắt phần answer ở trên)")

 
    # TEST 6 — Regression
    result = qa.ask(f"Hóa đơn của {company_a} có tổng tiền bao nhiêu?")
    print_case("TEST 6: regression (single-invoice, KHÔNG được là aggregate)",
               f"Hóa đơn của {company_a} có tổng tiền bao nhiêu?", result)
    assert result["intent"] == "total", (
        f"TEST 6 FAILED: câu hỏi 1 hóa đơn cụ thể bị nhận nhầm thành "
        f"'{result['intent']}' thay vì 'total'"
    )
    print("[PASS] TEST 6")
 
    print()
    print("=" * 80)
    print("TẤT CẢ TEST TÍCH HỢP ĐÃ PASS")
    print("=" * 80)
 
 
if __name__ == "__main__":
    main()
 