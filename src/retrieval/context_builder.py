from typing import Any, Dict, List


class InvoiceContextBuilder:

    """
    Convert retrieval results into a clean context
    that can later be sent to an LLM.
    """

    def build(
        self,
        query: str,
        results: List[
            Dict[str, Any]
        ],
    ) -> str:

        if not results:

            return (
                "Không tìm thấy hóa đơn "
                "phù hợp với câu hỏi."
            )

        sections = []

        sections.append(
            f"Câu hỏi: {query}"
        )

        sections.append(
            "Thông tin hóa đơn tìm được:"
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            company = metadata.get(
                "company",
                "",
            )

            date = metadata.get(
                "date",
                "",
            )

            address = metadata.get(
                "address",
                "",
            )

            total = metadata.get(
                "total",
                "",
            )

            section = (
                f"Hóa đơn {index}\n"
                f"- Company: {company}\n"
                f"- Date: {date}\n"
                f"- Address: {address}\n"
                f"- Total: {total}"
            )

            sections.append(
                section
            )

        return "\n\n".join(
            sections
        )

    def build_best_context(
        self,
        query: str,
        results: List[
            Dict[str, Any]
        ],
    ) -> str:

        if not results:

            return (
                "Không tìm thấy thông tin "
                "phù hợp."
            )

        best_result = results[0]

        metadata = best_result.get(
            "metadata",
            {},
        )

        company = metadata.get(
            "company",
            "",
        )

        date = metadata.get(
            "date",
            "",
        )

        address = metadata.get(
            "address",
            "",
        )

        total = metadata.get(
            "total",
            "",
        )

        return (
            f"Câu hỏi: {query}\n\n"
            f"Thông tin hóa đơn:\n"
            f"Company: {company}\n"
            f"Date: {date}\n"
            f"Address: {address}\n"
            f"Total: {total}"
        )

if __name__ == "__main__":

    builder = InvoiceContextBuilder()

    fake_results = [

        {
            "id": "iv01_inference",

            "distance": 0.9637,

            "semantic_score": 0.5092,

            "company_score": 1.0,

            "intent_score": 1.0,

            "final_score": 1.2092,

            "metadata": {
                "company":
                    "HOME MASTER HARDWARE& ELECTRICAL",

                "date":
                    "2017-12-22",

                "address":
                    "U13/EG BANDARSETIA ALAM, "
                    "40170 BANDARSETIA ALA, SELANGOR.",

                "total":
                    "15.9",
            },
        },

        {
            "id": "iv02_inference",

            "distance": 1.1836,

            "semantic_score": 0.4580,

            "company_score": 0.0,

            "intent_score": 1.0,

            "final_score": 0.6580,

            "metadata": {
                "company":
                    "LIGHTROOM GALLERY SDN BHD",

                "date":
                    "",

                "address":
                    "No:28,JALAN ASTANA 1C, "
                    "BANDAR BUKIT RAJA, "
                    "41050 KLANG SELANGOR D.E, MALAYSIA",

                "total":
                    "73.0",
            },
        },
    ]

    query = (
        "Hóa đơn của HOME MASTER HARDWARE "
        "có tổng tiền bao nhiêu?"
    )

    print()
    print("=" * 80)
    print("CONTEXT BUILDER TEST")
    print("=" * 80)

    print()
    print("FULL CONTEXT")
    print("-" * 80)

    print(
        builder.build(
            query,
            fake_results,
        )
    )

    print()
    print("BEST CONTEXT")
    print("-" * 80)

    print(
        builder.build_best_context(
            query,
            fake_results,
        )
    )

    print()
    print("=" * 80)