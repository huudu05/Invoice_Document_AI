import json
from pathlib import Path
 
from src.knowledge_base.chroma_store import ChromaInvoiceStore
 
 
# ============================================================
# CONFIGURATION
# ============================================================
 
INFERENCE_DIR = Path("outputs/inference")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "invoices"
 
 
# ============================================================
# LOAD JSON
# ============================================================
 
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    print()
    print("=" * 70)
    print("INVOICE KNOWLEDGE BASE INDEXING")
    print("=" * 70)
 
    # --------------------------------------------------------
    # Validate inference directory
    # --------------------------------------------------------
 
    if not INFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Inference directory not found: {INFERENCE_DIR}"
        )
 
    json_files = sorted(INFERENCE_DIR.glob("*_inference.json"))
 
    if not json_files:
        raise FileNotFoundError(
            f"No inference JSON files found in {INFERENCE_DIR}"
        )
 
    print(f"Found inference files: {len(json_files)}")
 
    # --------------------------------------------------------
    # Shared ChromaDB access (same class used by InvoiceRetriever
    # and run_pipeline.py) - single source of truth for how an
    # invoice is turned into a document + metadata.
    # --------------------------------------------------------
 
    store = ChromaInvoiceStore(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
 
    # --------------------------------------------------------
    # Index documents
    # --------------------------------------------------------
 
    indexed_count = 0
    skipped_count = 0
 
    for json_path in json_files:
        print()
        print(f"Processing: {json_path.name}")
 
        try:
            data = load_json(json_path)
        except (OSError, ValueError) as error:
            print(f"[WARNING] Could not read {json_path.name}: {error}. Skipping.")
            skipped_count += 1
            continue
 
        normalized = data.get("normalized")
 
        if not normalized:
            print(f"[WARNING] No 'normalized' field in {json_path.name}. Skipping.")
            skipped_count += 1
            continue
 
        invoice_id = json_path.stem
 
        try:
            store.add_invoice(invoice_id, normalized)
        except Exception as error:
            print(f"[WARNING] Failed to index {invoice_id}: {error}. Skipping.")
            skipped_count += 1
            continue
 
        indexed_count += 1
        print(f"[OK] Indexed: {invoice_id}")
 
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
 
    print()
    print("=" * 70)
    print("INDEXING COMPLETED")
    print("=" * 70)
    print(f"Documents indexed : {indexed_count}")
    print(f"Documents skipped : {skipped_count}")
    print(f"Collection        : {COLLECTION_NAME}")
    print(f"ChromaDB          : {CHROMA_DIR}")
    print(f"Total in store    : {store.count()}")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()