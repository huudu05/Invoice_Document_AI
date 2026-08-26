from pathlib import Path
import chromadb


class ChromaInvoiceStore:

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "invoices",
    ):
        self.persist_directory = Path(persist_directory)

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_invoice(
        self,
        invoice_id: str,
        normalized_invoice: dict
    ):

        document = self._build_document(
            normalized_invoice
        )

        metadata = {
            "company": normalized_invoice.get(
                "company", ""
            ),

            "date": normalized_invoice.get(
                "date", ""
            ),

            "address": normalized_invoice.get(
                "address", ""
            ),

            "total": float(
                normalized_invoice.get("total", 0)
                or 0
            )
        }

        self.collection.upsert(
            ids=[invoice_id],

            documents=[document],

            metadatas=[metadata]
        )

    @staticmethod
    def _build_document(invoice: dict) -> str:

        company = invoice.get("company", "")
        date = invoice.get("date", "")
        address = invoice.get("address", "")
        total = invoice.get("total", "")

        return (
            f"Invoice information. "
            f"Company: {company}. "
            f"Date: {date}. "
            f"Address: {address}. "
            f"Total: {total}."
        )

    def search(
        self,
        query: str,
        n_results: int = 5
    ):

        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    def count(self) -> int:
        return self.collection.count()

    def get_all_invoices(self) -> list[dict]:
        result = self.collection.get(include=["metadatas"])
        return result.get("metadatas", []) or []