from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_processor import OCRProcessor


manager = InputManager()

pages = manager.load("data/raw/invoice.pdf")

preprocessor = ImagePreprocessor()

pages = preprocessor.process(pages)

processor = OCRProcessor()

result = processor.recognize(pages)

print("=" * 50)
print("OCR RESULT")
print("=" * 50)

print(f"Number of pages: {len(result.pages)}")
"""
for page in result.pages:

    print(f"\nPage {page.page_number}")

    print(f"Detected words: {len(page.words)}")

    print("-" * 40)

    for word in page.words[:10]:

        print(
            f"{word.text:<30}"
            f"{word.confidence:.3f}"
        )
"""

"""
for page in result.pages:

    print("="*60)
    print(f"PAGE {page.page_number}")
    print("="*60)

    for word in page.words:
        print(word.text)
"""
"""
for page in result.pages:

    for word in page.words:

        print(
            word.text,
            word.bbox
        )
"""

for page in result.pages:

    print(f"\nPAGE {page.page_number}")

    for word in page.words[:10]:
        print(
            f"[{word.id:03d}] "
            f"{word.text:<35}"
            f"{word.confidence:.3f}"
        )