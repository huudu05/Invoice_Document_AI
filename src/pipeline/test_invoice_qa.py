from src.pipeline.invoice_qa import InvoiceQA

TEST_QUERIES = [

    "What is the total amount of the HOME MASTER HARDWARE invoice?",
    "What is the date of the HOME MASTER invoice?",
    "What is the address of HOME MASTER HARDWARE?",
    "What is the address of LIGHTROOM GALLERY?",
    "What is the total amount of the LIGHTROOM invoice?",
    "What is the date of the LIGHTROOM invoice?",
    "What is the total amount of the APPLE invoice?",
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
            metadata = source.get("metadata", {})

            print()
            print("SOURCE INVOICE")
            print("-" * 90)
            print(f"ID      : {source.get('id', '')}")
            print(f"Score   : {source.get('final_score', 0.0):.4f}")
            print(f"Company : {metadata.get('company', '')}")
            print(f"Date    : {metadata.get('date', '')}")
            print(f"Address : {metadata.get('address', '')}")
            print(f"Total   : {metadata.get('total', '')}")
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