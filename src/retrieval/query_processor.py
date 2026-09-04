import re
import calendar
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
 
 
@dataclass
class QueryInfo:
 
    original_query: str
 
    intent: str
 
    company: Optional[str]
 
    keywords: List[str]

    date_from: Optional[str] = None
    date_to: Optional[str] = None
    company_b: Optional[str] = None  # second company, only for "compare"
 
 
 
class QueryProcessor:
 
    TOTAL_KEYWORDS = [
        "total",
        "amount",
        "how much",
        "cost",
        "price",
        "sum",
    ]
 
    DATE_KEYWORDS = [
        "date",
        "invoice date",
        "when",
        "what day",
    ]
 
    ADDRESS_KEYWORDS = [
        "address",
        "located",
        "location",
        "where",
    ]
 
    COMPANY_KEYWORDS = [
        "company",
        "business",
        "vendor",
        "seller",
        "name of the company",
    ]
 
    AGGREGATE_SUM_KEYWORDS = [
        "total spending",
        "total expenses",
        "total of all",
        "sum of all",
        "overall spending",
        "all invoices total",
    ]
 
    AGGREGATE_COUNT_KEYWORDS = [
        "how many invoices",
        "number of invoices",
        "count invoices",
        "count the invoices",
    ]
 
    AGGREGATE_TOP_KEYWORDS = [
        "highest",
        "most",
        "largest",
        "lowest",
        "least",
        "top",
    ]
 
    AGGREGATE_TREND_KEYWORDS = [
        "by month",
        "each month",
        "trend",
        "over time",
        "monthly",
    ]
 
    AGGREGATE_COMPARE_KEYWORDS = [
        "compare",
        "vs",
        "versus",
        "compared to",
    ]
 
    MONTH_NAMES = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
 

    def process(self, query: str) -> QueryInfo:
 
        query = query.strip()
 
        if not query:
            raise ValueError("Query cannot be empty.")
 
        normalized_query = self._normalize_text(query)
 
        intent = self._detect_intent(normalized_query)

 
        # Aggregate intents have their own extraction logic.
        if intent.startswith("aggregate_"):
 
            date_from, date_to = self._extract_date_range(normalized_query)
            keywords = self._extract_keywords(normalized_query)
 
            if intent == "aggregate_compare":
 
                company_a, company_b = self._extract_two_companies(query)
 
                return QueryInfo(
                    original_query=query,
                    intent=intent,
                    company=company_a,
                    company_b=company_b,
                    keywords=keywords,
                    date_from=date_from,
                    date_to=date_to,
                )
 
            company = self._extract_company(query, normalized_query)
 
            return QueryInfo(
                original_query=query,
                intent=intent,
                company=company,
                keywords=keywords,
                date_from=date_from,
                date_to=date_to,
            )
 
        # Single-invoice intents: unchanged behavior.
        company = self._extract_company(query, normalized_query)
        keywords = self._extract_keywords(normalized_query)
 
        return QueryInfo(
            original_query=query,
            intent=intent,
            company=company,
            keywords=keywords,
        )
 

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()
 
    def _detect_intent(self, query: str) -> str:
 
        if self._contains_any(query, self.AGGREGATE_COMPARE_KEYWORDS):
            return "aggregate_compare"

        if self._contains_any(query, self.AGGREGATE_TOP_KEYWORDS):
            return "aggregate_top"

        if self._contains_any(query, self.AGGREGATE_TREND_KEYWORDS):
            return "aggregate_trend"

        if self._contains_any(query, self.AGGREGATE_COUNT_KEYWORDS):
            return "aggregate_count"

        if self._contains_any(query, self.AGGREGATE_SUM_KEYWORDS):
            return "aggregate_sum"
 
        if self._contains_any(query, self.TOTAL_KEYWORDS):
            return "total"
 
        if self._contains_any(query, self.DATE_KEYWORDS):
            return "date"
 
        if self._contains_any(query, self.ADDRESS_KEYWORDS):
            return "address"
 
        if self._contains_any(query, self.COMPANY_KEYWORDS):
            return "company"
 
        return "general"
 
 
    def _extract_date_range(self, query: str) -> Tuple[Optional[str], Optional[str]]:
 
        match = re.search(r"q(?:uarter)?\s*([1-4])\s+(\d{4})", query)
        if match:
            quarter = int(match.group(1))
            year = int(match.group(2))
            start_month = (quarter - 1) * 3 + 1
            end_month = start_month + 2
            end_day = calendar.monthrange(year, end_month)[1]
            return (
                f"{year:04d}-{start_month:02d}-01",
                f"{year:04d}-{end_month:02d}-{end_day:02d}",
            )
        
        month_pattern = "|".join(self.MONTH_NAMES.keys())
        match = re.search(rf"({month_pattern})\s+(\d{{4}})", query)
        if match:
            month = self.MONTH_NAMES[match.group(1)]
            year = int(match.group(2))
            end_day = calendar.monthrange(year, month)[1]
            return (
                f"{year:04d}-{month:02d}-01",
                f"{year:04d}-{month:02d}-{end_day:02d}",
            )
 
        # "3/2020" (month/year)
        match = re.search(r"\b(\d{1,2})/(\d{4})\b", query)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            if 1 <= month <= 12:
                end_day = calendar.monthrange(year, month)[1]
                return (
                    f"{year:04d}-{month:02d}-01",
                    f"{year:04d}-{month:02d}-{end_day:02d}",
                )
 
        # "2020" / "in 2020"
        match = re.search(r"\b(\d{4})\b", query)
        if match:
            year = int(match.group(1))
            return (f"{year:04d}-01-01", f"{year:04d}-12-31")
 
        return (None, None)

    def _extract_company(
        self,
        original_query: str,
        normalized_query: str,
    ) -> Optional[str]:
 
        patterns = [
            r"invoice\s+(?:for|from)\s+(.+?)(?:\s+has|\s+is|\s+on|\?|$)",
            r"(?:for|from)\s+(.+?)(?:\s+has|\s+is|\s+on|\?|$)",
            r"company\s+(.+?)(?:\s+has|\s+is|\?|$)",
            r"of\s+(?:the\s+)?(.+?)(?:\s+has|\s+is|\?|$)",
        ]
 
        for pattern in patterns:
            match = re.search(pattern, normalized_query, flags=re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                company = self._clean_company(company)
                if company and self._is_valid_company(company):
                    return company
                # invalid capture (e.g. "nào", "năm 2017") -> try next pattern
                continue
 
        return None

    _COMPANY_STOPWORDS = {
        "it", "this", "that", "what", "which", "who",
        "there", "here", "invoice", "invoice's",
    }
 
    _LEADING_STOPWORDS = {
        "in", "on", "at", "for", "with", "about",
        "the", "a", "an", "and", "or", "to",
    }
 
    def _is_valid_company(self, company: str) -> bool:
 
        if company.lower() in self._COMPANY_STOPWORDS:
            return False
 
        if re.fullmatch(r"year\s*\d{0,4}", company, flags=re.IGNORECASE):
            return False
 
        if re.fullmatch(r"\d+", company):
            return False
 
        if len(company) < 3:
            return False
 
        first_word = company.split()[0].lower()
        if first_word in self._LEADING_STOPWORDS:
            return False
 
        return True

 
    def _extract_two_companies(
        self,
        query: str,
    ) -> Tuple[Optional[str], Optional[str]]:
 
        normalized_query = self._normalize_text(query)
 
        pattern = (
            r"compare\s+(?:spending\s+(?:of|for)\s+)?(?:between\s+)?"
            r"(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)"
            r"(?:\s+in|\s+from|\?|$)"
        )
 
        match = re.search(pattern, normalized_query, flags=re.IGNORECASE)
 
        if not match:
            return None, None
 
        company_a = self._clean_company(match.group(1).strip())
        company_b = self._clean_company(match.group(2).strip())
 
        return (company_a or None), (company_b or None)
 
 
    def _clean_company(self, company: str) -> str:
 
        company = company.strip()
 
        company = re.sub(
            r"^(the\s+invoice\s+(?:of|from)|invoice\s+(?:of|from)|the\s+invoice|invoice)\s+",
            "",
            company,
            flags=re.IGNORECASE,
        )
 
        company = re.sub(
            r"^the\s+",
            "",
            company,
            flags=re.IGNORECASE,
        )
 
        company = re.sub(
            r"\s+(is|has|on|was|were|does|do)$",
            "",
            company,
            flags=re.IGNORECASE,
        )
 
        company = company.strip(" .,?!:")
 
        return company
 
    # KEYWORDS
    def _extract_keywords(self, query: str) -> List[str]:
 
        keywords = []
 
        groups = [
            self.TOTAL_KEYWORDS,
            self.DATE_KEYWORDS,
            self.ADDRESS_KEYWORDS,
            self.COMPANY_KEYWORDS,
            self.AGGREGATE_SUM_KEYWORDS,
            self.AGGREGATE_COUNT_KEYWORDS,
            self.AGGREGATE_TOP_KEYWORDS,
            self.AGGREGATE_TREND_KEYWORDS,
            self.AGGREGATE_COMPARE_KEYWORDS,
        ]
 
        for group in groups:
            for keyword in group:
                if keyword in query:
                    keywords.append(keyword)
 
        return list(dict.fromkeys(keywords))
 
 
    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)
 
 
def process_query(query: str) -> QueryInfo:
    processor = QueryProcessor()
    return processor.process(query)
 
 
if __name__ == "__main__":
 
    processor = QueryProcessor()
 
    test_queries = [
        "What is the total amount of the invoice from HOME MASTER HARDWARE?",
        "What is the date of the invoice from HOME MASTER?",
        "What was the total spending in December 2017?",
        "How many invoices are there from HOME MASTER?",
        "Which company has the highest total spending?",
        "Compare spending between HOME MASTER and LIGHTROOM",
        "Show monthly spending for 2017",
    ]
 
    print()
    print("=" * 80)
    print("QUERY PROCESSOR TEST (with aggregate intents)")
    print("=" * 80)
 
    for query in test_queries:
        result = processor.process(query)
 
        print()
        print(f"Query      : {result.original_query}")
        print(f"Intent     : {result.intent}")
        print(f"Company    : {result.company}")
        print(f"Company B  : {result.company_b}")
        print(f"Date range : {result.date_from} -> {result.date_to}")
        print(f"Keywords   : {result.keywords}")
 
    print()
    print("=" * 80)
 