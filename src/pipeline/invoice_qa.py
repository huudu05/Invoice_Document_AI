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


DEFAULT_TOP_K = 5

MIN_RELEVANCE_SCORE = 0.80


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

        self.retriever = (
            InvoiceRetriever()
        )

        self.context_builder = (
            InvoiceContextBuilder()
        )

        self.llm = (
            LLMGenerator()
            if use_llm
            else None
        )

        self.analytics = InvoiceAnalytics(
            store=self.retriever.store
        )


    def ask(
        self,
        query: str,
    ) -> Dict[str, Any]:

        if not query or not query.strip():

            return self._failure(
                query=query,
                answer=(
                    "Please enter a question."
                ),
            )

        query = query.strip()

        query_info = (
            self.retriever.query_processor.process(
                query
            )
        )

        if query_info.intent.startswith("aggregate_"):
            return self._handle_aggregate_query(query, query_info)

        results = (
            self.retriever.search(
                query=query,
                top_k=self.top_k,
            )
        )

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
                    "No invoice matching the question was found."
                ),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": None,
            }

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


        context = (
            self.context_builder.build_best_context(
                query=query,
                results=[
                    source
                ],
            )
        )


        if not self.use_llm:

            return {
                "success": True,
                "answer": (
                    "Relevant invoice information was found."
                ),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "context": context,
                "results": results,
                "source": source,
                "answer_source": "retrieval",
            }


        prompt = (
            self._build_prompt(
                query=query,
                context=context,
            )
        )


        try:

            answer = self.llm.generate(
                prompt
            )

        except Exception as error:

            return {
                "success": False,
                "answer": (
                    "Unable to generate an answer from the language model."
                ),
                "error": str(error),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": source,
            }

        if not answer:

            return {
                "success": False,
                "answer": (
                    "Unable to generate a suitable answer."
                ),
                "query": query,
                "intent": query_info.intent,
                "company": query_info.company,
                "results": results,
                "source": source,
            }

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

    def _handle_aggregate_query(
        self,
        query: str,
        query_info,
    ) -> Dict[str, Any]:

        intent = query_info.intent

        if not self.analytics.invoices:
            return {
                "success": False,
                "answer": "There are no invoices available for analysis.",
                "query": query,
                "intent": intent,
                "company": query_info.company,
                "source": None,
            }


        # Compute the exact numbers first
        if intent == "aggregate_compare":

            if not query_info.company or not query_info.company_b:
                return {
                    "success": False,
                    "answer": (
                        "Please specify the two companies to compare, for example: 'Compare spending between A and B'."
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
                f"Spending comparison:\n"
                f"- {data['company_a']['name']}: "
                f"total {data['company_a']['total']}, "
                f"{data['company_a']['count']} invoices, "
                f"average {data['company_a']['average']}\n"
                f"- {data['company_b']['name']}: "
                f"total {data['company_b']['total']}, "
                f"{data['company_b']['count']} invoices, "
                f"average {data['company_b']['average']}"
            )
            structured_data = data

        elif intent == "aggregate_top":

            top = self.analytics.top_companies(n=5)
            if not top:
                summary = "No spending data available by company."
            else:
                lines = ["Spending ranking by company:"]
                for rank, (company, amount) in enumerate(top, start=1):
                    lines.append(f"{rank}. {company}: {amount}")
                summary = "\n".join(lines)
            structured_data = {"ranking": top}

        elif intent == "aggregate_trend":

            trend = self.analytics.spending_by_month(company=query_info.company)
            if not trend:
                summary = "No monthly spending data available."
            else:
                lines = ["Monthly spending:"]
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

        else:  # aggregate_sum

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

        # If LLM disabled: return the raw summary directly.

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

        prompt = f"""
You are an AI assistant answering invoice spending analysis questions.

The following results have already been calculated accurately by the system.
Only express these numbers naturally in your answer.
Do not recalculate anything or invent additional figures.

DATA:
{summary}

QUESTION:
{query}

ANSWER:
""".strip()

        try:
            answer = self.llm.generate(prompt)
        except Exception as error:
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

        for result in results:
            final_score = float(
                result.get(
                    "final_score",
                    0.0,
                )
            )

            # Minimum relevance
            if (final_score < self.min_relevance_score):
                continue

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
                "this invoice",
            )
        )

        intent = (
            query_info.intent
        )

        if intent == "total":
            if not self._has_value(
                metadata.get(
                    "total"
                )
            ):

                return (
                    f"The total amount for invoice {company} was not found in the data."
                )

        elif intent == "date":

            if not self._has_value(
                metadata.get(
                    "date"
                )
            ):
                return (
                    f"The date for invoice {company} was not found in the data."
                )

        elif intent == "address":

            if not self._has_value(
                metadata.get(
                    "address"
                )
            ):

                return (
                    f"The address for invoice {company} was not found in the data."
                )

        elif intent == "company":

            if not self._has_value(
                metadata.get(
                    "company"
                )
            ):

                return (
                    f"The company name for invoice {company} was not found in the data."
                )

        return None

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

    def _build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:

        return f"""
You are an AI assistant specialized in answering business invoice questions.

Use only the information provided in the CONTEXT.

RULES:
1. Do not use knowledge outside the CONTEXT.
2. Do not make assumptions or invent information.
3. If the CONTEXT does not contain the required information, clearly state that it was not found.
4. Answer directly and concisely.
5. Preserve amounts, dates, and addresses exactly as provided.
6. Do not explain your reasoning.
7. Answer only the current question.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


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


    @staticmethod
    def _has_value(
        value: Any,
    ) -> bool:

        if value is None:

            return False

        return bool(
            str(value).strip()
        )



def ask_invoice(
    query: str,
) -> Dict[str, Any]:

    qa = InvoiceQA()

    return qa.ask(
        query
    )


if __name__ == "__main__":

    test_queries = [
        "What is the total amount of the HOME MASTER HARDWARE invoice?",
        "What is the date of the HOME MASTER invoice?",
        "What is the address of HOME MASTER HARDWARE?",
        "What is the address of LIGHTROOM GALLERY?",
        "What is the total amount of the LIGHTROOM invoice?",
        "What is the date of the LIGHTROOM invoice?",
        "What is the total amount of the APPLE invoice?",
    ]

    qa = InvoiceQA()

    print()
    print("=" * 100)
    print("INVOICE QUESTION ANSWERING TEST")
    print("=" * 100)

    for query in test_queries:

        print()
        print("=" * 100)
        print(f"QUERY:\n{query}")
        result = qa.ask(
            query
        )

        print()
        print("QUERY ANALYSIS")
        print(
            f"Intent  : "
            f"{result.get('intent', '')}"
        )
        print(
            f"Company : "
            f"{result.get('company', '')}"
        )

        print()
        print("ANSWER")
        print("-" * 80)
        print(result["answer"])

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
        print("SOURCE INVOICE")
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
            print("No valid source.")
    print()
    print("=" * 100)
    print("TEST COMPLETED")
    print("=" * 100)