from typing import Any, Dict, List, Optional

from src.retrieval.invoice_retriever import (
    InvoiceRetriever,
)

from src.retrieval.context_builder import (
    InvoiceContextBuilder,
)

from src.generation.llm_generator import (
    LLMGenerator,
)
from src.analytics.invoice_analytics import InvoiceAnalytics


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5

MIN_RELEVANCE_SCORE = 0.80


# ============================================================
# INVOICE QA
# ============================================================

class InvoiceQA:

    """
    End-to-end invoice question answering pipeline.

    Pipeline:

        Query
          ↓
        QueryProcessor
          ↓
        Retrieval
          ↓
        Relevance Gate
          ↓
        Field Validation
          ↓
        Direct Answer / Context
          ↓
        Gemini
          ↓
        Final Answer
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
        use_llm: bool = True,
    ):

        self.top_k = top_k

        self.min_relevance_score = (
            min_relevance_score
        )

        self.use_llm = use_llm

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        self.retriever = (
            InvoiceRetriever()
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        self.context_builder = (
            InvoiceContextBuilder()
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.llm = (
            LLMGenerator()
            if use_llm
            else None
        )

        self.analytics = InvoiceAnalytics(
            store=self.retriever.store
        )

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        query: str,
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not query or not query.strip():

            return self._failure(
                query=query,
                answer=(
                    "Vui lòng nhập câu hỏi."
                ),
            )

        query = query.strip()

        # ----------------------------------------------------
        # Query analysis
        # ----------------------------------------------------

        query_info = (
            self.retriever.query_processor.process(
                query
            )
        )
        # ----------------------------------------------------
        # NEW: Aggregate queries (phân tích tổng hợp)
        #
        # Các intent này không tra 1 hóa đơn cụ thể, nên tách khỏi
        # luồng retrieval/relevance-gate/field-validation bên dưới
        # (vốn được thiết kế cho single-invoice lookup).
        # ----------------------------------------------------

        if query_info.intent.startswith("aggregate_"):
            return self._handle_aggregate_query(query, query_info)

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        results = (
            self.retriever.search(
                query=query,
                top_k=self.top_k,
            )
        )

        # ----------------------------------------------------
        # Relevance Gate
        # ----------------------------------------------------

        source = (
            self._select_relevant_result(
                query_info=query_info,
                results=results,
            )
        )

        if source is None:

            return {
                "success": False,
                "answer": (
                    "Không tìm thấy hóa đơn "
                    "phù hợp với câu hỏi."
                ),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": None,
            }

        # ----------------------------------------------------
        # Validate requested field
        # ----------------------------------------------------

        field_error = (
            self._validate_requested_field(
                query_info=query_info,
                result=source,
            )
        )

        if field_error:

            return {
                "success": False,
                "answer": field_error,
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": source,
            }

        # ----------------------------------------------------
        # Direct answer
        #
        # For simple structured fields:
        #
        # total
        # date
        # address
        # company
        #
        # We can answer directly.
        # ----------------------------------------------------

        direct_answer = (
            self._build_direct_answer(
                query_info=query_info,
                result=source,
            )
        )

        if direct_answer is not None:

            return {
                "success": True,
                "answer": direct_answer,
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": source,
                "answer_source": "structured_data",
            }

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = (
            self.context_builder.build_best_context(
                query=query,
                results=[
                    source
                ],
            )
        )

        # ----------------------------------------------------
        # If LLM disabled
        # ----------------------------------------------------

        if not self.use_llm:

            return {
                "success": True,
                "answer": (
                    "Đã tìm thấy thông tin "
                    "hóa đơn phù hợp."
                ),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "context": context,
                "results": results,
                "source": source,
                "answer_source": "retrieval",
            }

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = (
            self._build_prompt(
                query=query,
                context=context,
            )
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        try:

            answer = self.llm.generate(
                prompt
            )

        except Exception as error:

            return {
                "success": False,
                "answer": (
                    "Không thể tạo câu trả lời "
                    "từ mô hình ngôn ngữ."
                ),
                "error": str(error),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": source,
            }

        # ----------------------------------------------------
        # Empty answer
        # ----------------------------------------------------

        if not answer:

            return {
                "success": False,
                "answer": (
                    "Không thể tạo câu trả lời "
                    "phù hợp."
                ),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": source,
            }

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        return {
            "success": True,
            "answer": answer,
            "query": query,
            "intent": query_info.intent,
            "company": query_info.company,
            "context": context,
            "results": results,
            "source": source,
            "answer_source": "llm",
        }

        # ========================================================
    # AGGREGATE QUERY (phân tích tổng hợp)
    # ========================================================

    def _handle_aggregate_query(
        self,
        query: str,
        query_info,
    ) -> Dict[str, Any]:

        intent = query_info.intent

        # ----------------------------------------------------
        # Guard: no invoice indexed at all
        # ----------------------------------------------------

        if not self.analytics.invoices:
            return {
                "success": False,
                "answer": "Chưa có hóa đơn nào trong hệ thống để phân tích.",
                "query": query,
                "intent": intent,
                "company": query_info.company,
                "source": None,
            }

        # ----------------------------------------------------
        # Compute the exact numbers first (pure code, no LLM).
        # This is the single source of truth for the answer.
        # ----------------------------------------------------

        if intent == "aggregate_compare":

            if not query_info.company or not query_info.company_b:
                return {
                    "success": False,
                    "answer": (
                        "Vui lòng nêu rõ 2 công ty cần so sánh, "
                        "ví dụ: 'So sánh chi tiêu giữa A và B'."
                    ),
                    "query": query,
                    "intent": intent,
                    "company": query_info.company,
                    "source": None,
                }

            data = self.analytics.compare_companies(
                query_info.company, query_info.company_b
            )
            summary = (
                f"So sánh chi tiêu:\n"
                f"- {data['company_a']['name']}: "
                f"tổng {data['company_a']['total']}, "
                f"{data['company_a']['count']} hóa đơn, "
                f"trung bình {data['company_a']['average']}\n"
                f"- {data['company_b']['name']}: "
                f"tổng {data['company_b']['total']}, "
                f"{data['company_b']['count']} hóa đơn, "
                f"trung bình {data['company_b']['average']}"
            )
            structured_data = data

        elif intent == "aggregate_top":

            top = self.analytics.top_companies(n=5)
            if not top:
                summary = "Không có dữ liệu chi tiêu theo công ty."
            else:
                lines = ["Xếp hạng chi tiêu theo công ty:"]
                for rank, (company, amount) in enumerate(top, start=1):
                    lines.append(f"{rank}. {company}: {amount}")
                summary = "\n".join(lines)
            structured_data = {"ranking": top}

        elif intent == "aggregate_trend":

            trend = self.analytics.spending_by_month(company=query_info.company)
            if not trend:
                summary = "Không có dữ liệu để thống kê theo tháng."
            else:
                lines = ["Chi tiêu theo tháng:"]
                for month, amount in trend.items():
                    lines.append(f"{month}: {amount}")
                summary = "\n".join(lines)
            structured_data = {"by_month": trend}

        elif intent == "aggregate_count":

            count = self.analytics.count_invoices(
                company=query_info.company,
                date_from=query_info.date_from,
                date_to=query_info.date_to,
            )
            summary = self.analytics.summary_text(
                company=query_info.company,
                date_from=query_info.date_from,
                date_to=query_info.date_to,
            )
            structured_data = {"count": count}

        else:  # aggregate_sum (default)

            summary = self.analytics.summary_text(
                company=query_info.company,
                date_from=query_info.date_from,
                date_to=query_info.date_to,
            )
            structured_data = {
                "total": self.analytics.total_spending(
                    company=query_info.company,
                    date_from=query_info.date_from,
                    date_to=query_info.date_to,
                )
            }

        # ----------------------------------------------------
        # If LLM disabled: return the raw summary directly.
        # ----------------------------------------------------

        if not self.use_llm:
            return {
                "success": True,
                "answer": summary,
                "query": query,
                "intent": intent,
                "company": query_info.company,
                "data": structured_data,
                "answer_source": "analytics",
            }

        # ----------------------------------------------------
        # Let the LLM phrase the already-computed numbers as a
        # natural sentence. It must NOT recompute anything.
        # ----------------------------------------------------

        prompt = f"""
Bạn là trợ lý AI trả lời câu hỏi phân tích chi tiêu hóa đơn.

Dưới đây là kết quả đã được HỆ THỐNG TÍNH TOÁN CHÍNH XÁC sẵn.
CHỈ diễn đạt lại các con số này thành câu trả lời tự nhiên,
KHÔNG được tự tính toán lại, KHÔNG bịa thêm số liệu nào khác.

DỮ LIỆU:
{summary}

CÂU HỎI:
{query}

CÂU TRẢ LỜI:
""".strip()

        try:
            answer = self.llm.generate(prompt)
        except Exception as error:
            # Fallback: LLM lỗi vẫn trả lời được bằng summary thô,
            # vì con số đã có sẵn, không phụ thuộc LLM để đúng.
            return {
                "success": True,
                "answer": summary,
                "query": query,
                "intent": intent,
                "company": query_info.company,
                "data": structured_data,
                "answer_source": "analytics_fallback",
                "error": str(error),
            }

        if not answer:
            answer = summary

        return {
            "success": True,
            "answer": answer,
            "query": query,
            "intent": intent,
            "company": query_info.company,
            "data": structured_data,
            "answer_source": "analytics_llm",
        }

    # ========================================================
    # SELECT RELEVANT RESULT
    # ========================================================

    def _select_relevant_result(
        self,
        query_info,
        results: List[
            Dict[str, Any]
        ],
    ) -> Optional[
        Dict[str, Any]
    ]:

        if not results:

            return None

        # ----------------------------------------------------
        # Results are already reranked.
        # ----------------------------------------------------

        for result in results:

            final_score = float(
                result.get(
                    "final_score",
                    0.0,
                )
            )

            # ------------------------------------------------
            # Minimum relevance
            # ------------------------------------------------

            if (
                final_score
                < self.min_relevance_score
            ):

                continue

            # ------------------------------------------------
            # If company specified,
            # require company match.
            # ------------------------------------------------

            if query_info.company:

                company_score = float(
                    result.get(
                        "company_score",
                        0.0,
                    )
                )

                if company_score <= 0:

                    continue

            return result

        return None

    # ========================================================
    # VALIDATE REQUESTED FIELD
    # ========================================================

    def _validate_requested_field(
        self,
        query_info,
        result: Dict[str, Any],
    ) -> Optional[str]:

        metadata = (
            result.get(
                "metadata",
                {},
            )
        )

        company = (
            metadata.get(
                "company",
                "hóa đơn này",
            )
        )

        intent = (
            query_info.intent
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        if intent == "total":

            if not self._has_value(
                metadata.get(
                    "total"
                )
            ):

                return (
                    f"Không tìm thấy tổng tiền "
                    f"của hóa đơn {company} "
                    f"trong dữ liệu."
                )

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        elif intent == "date":

            if not self._has_value(
                metadata.get(
                    "date"
                )
            ):

                return (
                    f"Không tìm thấy ngày "
                    f"của hóa đơn {company} "
                    f"trong dữ liệu."
                )

        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        elif intent == "address":

            if not self._has_value(
                metadata.get(
                    "address"
                )
            ):

                return (
                    f"Không tìm thấy địa chỉ "
                    f"của hóa đơn {company} "
                    f"trong dữ liệu."
                )

        # ----------------------------------------------------
        # Company
        # ----------------------------------------------------

        elif intent == "company":

            if not self._has_value(
                metadata.get(
                    "company"
                )
            ):

                return (
                    "Không tìm thấy tên công ty "
                    "trong dữ liệu hóa đơn."
                )

        return None

    # ========================================================
    # DIRECT ANSWER
    # ========================================================

    def _build_direct_answer(
        self,
        query_info,
        result: Dict[str, Any],
    ) -> Optional[str]:

        metadata = (
            result.get(
                "metadata",
                {},
            )
        )

        intent = (
            query_info.intent
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        if intent == "total":

            total = metadata.get(
                "total",
                "",
            )

            if self._has_value(
                total
            ):

                return str(
                    total
                ).strip()

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        if intent == "date":

            date = metadata.get(
                "date",
                "",
            )

            if self._has_value(
                date
            ):

                return str(
                    date
                ).strip()

        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        if intent == "address":

            address = metadata.get(
                "address",
                "",
            )

            if self._has_value(
                address
            ):

                return str(
                    address
                ).strip()

        # ----------------------------------------------------
        # Company
        # ----------------------------------------------------

        if intent == "company":

            company = metadata.get(
                "company",
                "",
            )

            if self._has_value(
                company
            ):

                return str(
                    company
                ).strip()

        return None

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def _build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:

        return f"""
Bạn là trợ lý AI chuyên trả lời câu hỏi
về hóa đơn doanh nghiệp.

CHỈ sử dụng thông tin có trong CONTEXT.

QUY TẮC:

1. Không được sử dụng kiến thức bên ngoài CONTEXT.
2. Không được tự suy đoán hoặc bịa thông tin.
3. Nếu CONTEXT không chứa thông tin cần thiết,
   hãy nói rõ rằng không tìm thấy thông tin trong dữ liệu.
4. Trả lời trực tiếp và ngắn gọn.
5. Giữ nguyên số tiền, ngày tháng và địa chỉ.
6. Không giải thích quá trình suy luận.
7. Chỉ trả lời câu hỏi hiện tại.

CONTEXT:
{context}

CÂU HỎI:
{query}

CÂU TRẢ LỜI:
""".strip()

    # ========================================================
    # FAILURE
    # ========================================================

    @staticmethod
    def _failure(
        query: str,
        answer: str,
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "answer": answer,
            "query": query,
            "results": [],
            "source": None,
        }

    # ========================================================
    # HAS VALUE
    # ========================================================

    @staticmethod
    def _has_value(
        value: Any,
    ) -> bool:

        if value is None:

            return False

        return bool(
            str(value).strip()
        )


# ============================================================
# HELPER
# ============================================================

def ask_invoice(
    query: str,
) -> Dict[str, Any]:

    qa = InvoiceQA()

    return qa.ask(
        query
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "Hóa đơn của HOME MASTER HARDWARE có tổng tiền bao nhiêu?",

        "Ngày của hóa đơn HOME MASTER là ngày nào?",

        "Địa chỉ của HOME MASTER HARDWARE là gì?",

        "Địa chỉ của LIGHTROOM GALLERY là gì?",

        "Tổng tiền hóa đơn LIGHTROOM là bao nhiêu?",

        "Ngày của hóa đơn LIGHTROOM là ngày nào?",

        "Tổng tiền hóa đơn APPLE là bao nhiêu?",
    ]

    qa = InvoiceQA()

    print()
    print("=" * 100)
    print("INVOICE QUESTION ANSWERING TEST")
    print("=" * 100)

    for query in test_queries:

        print()
        print("=" * 100)

        print(
            f"QUERY:\n{query}"
        )

        result = qa.ask(
            query
        )

        print()
        print(
            "QUERY ANALYSIS"
        )

        print(
            f"Intent  : "
            f"{result.get('intent', '')}"
        )

        print(
            f"Company : "
            f"{result.get('company', '')}"
        )

        print()
        print(
            "ANSWER"
        )

        print("-" * 80)

        print(
            result["answer"]
        )

        print()
        print(
            f"Success : "
            f"{result['success']}"
        )

        print(
            f"Answer Source : "
            f"{result.get('answer_source', '')}"
        )

        print()
        print(
            "SOURCE INVOICE"
        )

        print("-" * 80)

        source = (
            result.get(
                "source"
            )
        )

        if source:

            metadata = (
                source.get(
                    "metadata",
                    {},
                )
            )

            print(
                f"ID      : "
                f"{source.get('id', '')}"
            )

            print(
                f"Score   : "
                f"{source.get('final_score', 0.0):.4f}"
            )

            print(
                f"Company : "
                f"{metadata.get('company', '')}"
            )

            print(
                f"Date    : "
                f"{metadata.get('date', '')}"
            )

            print(
                f"Address : "
                f"{metadata.get('address', '')}"
            )

            print(
                f"Total   : "
                f"{metadata.get('total', '')}"
            )

        else:

            print(
                "No valid source."
            )

    print()
    print("=" * 100)
    print("TEST COMPLETED")
    print("=" * 100)