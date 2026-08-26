from src.retrieval.invoice_retriever import InvoiceRetriever


retriever = InvoiceRetriever()

queries = [
    "Hóa đơn của HOME MASTER HARDWARE có tổng tiền bao nhiêu?",
    "Ngày của hóa đơn HOME MASTER là ngày nào?",
    "Địa chỉ của công ty trên hóa đơn là gì?"
]

for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:")
    print(query)

    result = retriever.retrieve(
        query=query,
        top_k=3
    )

    documents = result["documents"][0]
    distances = result["distances"][0]

    for i, (document, distance) in enumerate(
        zip(documents, distances),
        start=1
    ):

        print(f"\nResult {i}")
        print(f"Distance: {distance:.4f}")
        print(document)