from src.processing.invoice_normalizer import InvoiceNormalizer


data = {
    "company": "HOME MASTER HARDWARE& ELECTRICAL",
    "date": "22/12/201714:03",
    "address": "U13/EG BANDARSETIA ALAM, 40170 BANDARSETIA ALA, SELANGOR.",
    "total": "15.90"
}


normalizer = InvoiceNormalizer()

result = normalizer.normalize(data)

print("=" * 60)
print("NORMALIZED INVOICE")
print("=" * 60)

for key, value in result.items():
    print(f"{key:10}: {value}")