from src.pipeline.invoice_qa import InvoiceQA

TEST_QUERIES = [

    "Hóa đơn của HOME MASTER HARDWARE có tổng tiền bao nhiêu?",

    "Ngày của hóa đơn HOME MASTER là ngày nào?",

    "Địa chỉ của HOME MASTER HARDWARE là gì?",

    "Địa chỉ của LIGHTROOM GALLERY là gì?",

    "Tổng tiền hóa đơn LIGHTROOM là bao nhiêu?",

    "Ngày của hóa đơn LIGHTROOM là ngày nào?",

    "Tổng tiền hóa đơn APPLE là bao nhiêu?",
]

def main():

    qa = InvoiceQA(
        top_k=5,
        min_relevance_score=0.80,
    )

    print()
    print("=" * 90)
    print("INVOICE QUESTION ANSWERING TEST")
    print("=" * 90)

    for query in TEST_QUERIES:

        print()
        print("=" * 90)
        print(f"QUERY:\n{query}")
        result = qa.ask(query)

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
        print("-" * 90)
        print(
            result.get(
                "answer",
                "",
            )
        )

        print()
        print(
            f"Success : "
            f"{result.get('success', False)}"
        )

        source = result.get(
            "source"
        )

        if source:
            metadata = source.get(
                "metadata",
                {},
            )
            print()
            print("SOURCE INVOICE")
            print("-" * 90)
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
            print()
            print("SOURCE INVOICE")
            print("-" * 90)
            print("No valid source.")

    print()
    print("=" * 90)
    print("TEST COMPLETED")
    print("=" * 90)

if __name__ == "__main__":
    main()