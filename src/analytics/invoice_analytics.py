from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
 
from src.knowledge_base.chroma_store import ChromaInvoiceStore
 
 
class InvoiceAnalytics:
    """
    Performs analytics on invoice metadata dictionaries
    (company/date/address/total).

    The `date` field is normalized as "YYYY-MM-DD" (ISO),
    so direct string comparison preserves chronological order.
    """
 
    def __init__(self, store: Optional[ChromaInvoiceStore] = None):
        self.store = store or ChromaInvoiceStore()
        self.invoices: List[Dict[str, Any]] = self.store.get_all_invoices()
 
    def refresh(self) -> None:
        """Refresh the invoice list after new invoices are indexed."""
        self.invoices = self.store.get_all_invoices()

    def _filter(
        self,
        company: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
 
        results = self.invoices
 
        if company:
            company_lower = company.strip().lower()
            results = [
                invoice for invoice in results
                if company_lower in str(invoice.get("company", "")).lower()
                or str(invoice.get("company", "")).lower() in company_lower
            ]
 
        if date_from:
            results = [
                invoice for invoice in results
                if self._has_date(invoice) and invoice["date"] >= date_from
            ]
 
        if date_to:
            results = [
                invoice for invoice in results
                if self._has_date(invoice) and invoice["date"] <= date_to
            ]
 
        return results
 
    @staticmethod
    def _has_date(invoice: Dict[str, Any]) -> bool:
        date = invoice.get("date")
        return bool(date and str(date).strip())
 
    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
 
    def total_spending(
        self,
        company: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> float:
 
        filtered = self._filter(company, date_from, date_to)
        return round(
            sum(self._to_float(invoice.get("total")) for invoice in filtered),
            2,
        )
 
    def count_invoices(
        self,
        company: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> int:
 
        return len(self._filter(company, date_from, date_to))
 
    def average_amount(
        self,
        company: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> float:
 
        filtered = self._filter(company, date_from, date_to)
        if not filtered:
            return 0.0
 
        total = sum(self._to_float(invoice.get("total")) for invoice in filtered)
        return round(total / len(filtered), 2)
 
    def spending_by_company(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, float]:
 
        filtered = self._filter(None, date_from, date_to)
        totals: Dict[str, float] = defaultdict(float)
 
        for invoice in filtered:
            company = str(invoice.get("company", "")).strip() or "(Unknown)"
            totals[company] += self._to_float(invoice.get("total"))
 
        return {
            company: round(amount, 2)
            for company, amount in sorted(
                totals.items(), key=lambda item: item[1], reverse=True
            )
        }
 
    def spending_by_month(
        self,
        company: Optional[str] = None,
    ) -> Dict[str, float]:
 
        filtered = self._filter(company)
        totals: Dict[str, float] = defaultdict(float)
 
        for invoice in filtered:
            if not self._has_date(invoice):
                continue
            month_key = str(invoice["date"])[:7]  # "YYYY-MM"
            totals[month_key] += self._to_float(invoice.get("total"))
 
        return dict(sorted(totals.items()))
 
    def top_companies(self, n: int = 5) -> List[Tuple[str, float]]:
        by_company = self.spending_by_company()
        return list(by_company.items())[:n]
 
    def compare_companies(
        self,
        company_a: str,
        company_b: str,
    ) -> Dict[str, Any]:
 
        return {
            "company_a": {
                "name": company_a,
                "total": self.total_spending(company=company_a),
                "count": self.count_invoices(company=company_a),
                "average": self.average_amount(company=company_a),
            },
            "company_b": {
                "name": company_b,
                "total": self.total_spending(company=company_b),
                "count": self.count_invoices(company=company_b),
                "average": self.average_amount(company=company_b),
            },
        }
 
    def summary_text(
        self,
        company: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
 
        total = self.total_spending(company, date_from, date_to)
        count = self.count_invoices(company, date_from, date_to)
        average = self.average_amount(company, date_from, date_to)
 
        lines = ["Summary results (calculated accurately by the system):"]
        if company:
            lines.append(f"- Company: {company}")
        if date_from or date_to:
            lines.append(f"- Date range: {date_from or '...'} to {date_to or '...'}")
        lines.append(f"- Number of invoices: {count}")
        lines.append(f"- Total spending: {total}")
        lines.append(f"- Average per invoice: {average}")
 
        return "\n".join(lines)
 
 
if __name__ == "__main__":
    analytics = InvoiceAnalytics()
 
    print("=" * 70)
    print("INVOICE ANALYTICS TEST")
    print("=" * 70)
 
    print(f"\nTotal invoices     : {len(analytics.invoices)}")
    print(f"Total spending     : {analytics.total_spending()}")
    print(f"Average per invoice: {analytics.average_amount()}")
 
    print("\nSpending by company:")
    for company, amount in analytics.spending_by_company().items():
        print(f"  {company:<40} {amount}")
 
    print("\nSpending by month:")
    for month, amount in analytics.spending_by_month().items():
        print(f"  {month:<10} {amount}")
 