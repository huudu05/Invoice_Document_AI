import re
import unicodedata
 
from pathlib import Path
from typing import Any, Dict, List, Optional
 
from src.knowledge_base.chroma_store import ChromaInvoiceStore
from src.retrieval.query_processor import QueryInfo, QueryProcessor
 
 
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "invoices"
 
SEMANTIC_WEIGHT = 1.00
COMPANY_WEIGHT = 0.50

FIELD_AVAILABILITY_WEIGHT = 0.30
 
DEFAULT_CANDIDATE_MULTIPLIER = 2
MIN_CANDIDATE_COUNT = 10
 
 
class InvoiceRetriever:
 
    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.query_processor = QueryProcessor()

 
        # Shared Chroma access
        self.store = ChromaInvoiceStore(
            persist_directory=str(chroma_dir),
            collection_name=collection_name,
        )
 
        if self.store.count() == 0:
            raise RuntimeError(
                f"ChromaDB collection '{collection_name}' is empty. "
                f"Hay chay index_invoice.py truoc."
            )
 
 
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
 
        if top_k <= 0:
            return []
 
        query = query.strip()
 
        query_info = self.query_processor.process(query)
 
        collection_count = self.store.count()
        if collection_count <= 0:
            return []
 
        candidate_k = max(
            top_k * DEFAULT_CANDIDATE_MULTIPLIER,
            MIN_CANDIDATE_COUNT,
        )
        candidate_k = min(candidate_k, collection_count)
 
        results = self.store.search(query, n_results=candidate_k)
 
        candidates = self._convert_results(results)
        if not candidates:
            return []
 
        ranked_results = self._rerank(candidates, query_info)
 
        for result in ranked_results:
            result["query_info"] = {
                "intent": query_info.intent,
                "company": query_info.company,
                "keywords": query_info.keywords,
            }
 
        return ranked_results[:top_k]
 
 
    @staticmethod
    def _convert_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
 
        candidates = []
        for index, document_id in enumerate(ids):
            document = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = (
                float(distances[index]) if index < len(distances) else 999.0
            )
 
            candidates.append({
                "id": document_id,
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            })
 
        return candidates
 
    def _rerank(
        self,
        candidates: List[Dict[str, Any]],
        query_info: QueryInfo,
    ) -> List[Dict[str, Any]]:
 
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            distance = float(candidate.get("distance", 999.0))
 
            semantic_score = 1.0 / (1.0 + distance)
            company_score = self._company_match_score(
                query_info.company, metadata
            )

            field_availability_score = self._field_availability_score(
                query_info.intent, metadata
            )
 
            final_score = (
                SEMANTIC_WEIGHT * semantic_score
                + COMPANY_WEIGHT * company_score
                + FIELD_AVAILABILITY_WEIGHT * field_availability_score
            )
 
            has_company_query = bool(query_info.company)
            is_company_match = company_score > 0.0

            is_relevant = (
                is_company_match if has_company_query else True
            )
 
            candidate["semantic_score"] = semantic_score
            candidate["company_score"] = company_score
            candidate["field_availability_score"] = field_availability_score
            candidate["final_score"] = final_score
            candidate["is_company_match"] = is_company_match
            candidate["is_relevant"] = is_relevant
 
        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        return candidates
 
 
    def _company_match_score(
        self,
        query_company: Optional[str],
        metadata: Dict[str, Any],
    ) -> float:
 
        if not query_company:
            return 0.0
 
        document_company = metadata.get("company", "")
        if not document_company:
            return 0.0
 
        query_company = self._normalize_text(query_company)
        document_company = self._normalize_text(document_company)
 
        if not query_company or not document_company:
            return 0.0
 
        if query_company == document_company:
            return 1.0
 
        if query_company in document_company:
            return 1.0
 
        if document_company in query_company:
            return 1.0
 
        query_tokens = set(query_company.split())
        document_tokens = set(document_company.split())
 
        if not query_tokens:
            return 0.0
 
        overlap = query_tokens & document_tokens
        if not overlap:
            return 0.0
 
        token_score = len(overlap) / len(query_tokens)
        return min(token_score, 1.0)
 
 
    def _field_availability_score(
        self,
        intent: str,
        metadata: Dict[str, Any],
    ) -> float:
 
        field_map = {
            "total": "total",
            "date": "date",
            "address": "address",
            "company": "company",
        }
 
        field_name = field_map.get(intent)
        if not field_name:
            return 0.0
 
        return 1.0 if self._has_value(metadata.get(field_name)) else 0.0
 
 
    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
 
        value = str(value).lower().strip()
        if not value:
            return ""
 
        value = unicodedata.normalize("NFKD", value)
        value = "".join(
            character for character in value
            if not unicodedata.combining(character)
        )
 
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value)
 
        return value.strip()
 
 
    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        return bool(str(value).strip())

def search_invoice(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    retriever = InvoiceRetriever()
    return retriever.search(query=query, top_k=top_k)
 
 
if __name__ == "__main__":
    retriever = InvoiceRetriever()
 
    test_queries = [
        "What is the total amount of the HOME MASTER HARDWARE invoice?",
        "What is the address of LIGHTROOM GALLERY?",
        "What is the total amount of the APPLE invoice?",
    ]
 
    for query in test_queries:
        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
 
        results = retriever.search(query=query, top_k=3)
        if not results:
            print("No results.")
            continue
 
        for i, result in enumerate(results, start=1):
            metadata = result["metadata"]
            print(f"\nResult {i} | final_score={result['final_score']:.4f} "
                  f"| company={metadata.get('company', '')}")
 