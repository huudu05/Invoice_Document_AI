from pathlib import Path
from typing import List

from PIL import Image
from pdf2image import convert_from_path

class PdfLoader:
    """
    Convert PDF to a list of PIL images.
    """

    def __init__(self, dpi: int = 300, poppler_path: str | None = None):
        self.dpi = dpi
        self.poppler_path = poppler_path


    def load(self, pdf_path: str | Path) -> List[Image.Image]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            images = convert_from_path(pdf_path, dpi=self.dpi, poppler_path=self.poppler_path)
            return images
        except Exception as e:
            raise RuntimeError(f"Unable to load PDF: {pdf_path}") from e

if __name__ == "__main__":
    loader = PdfLoader(poppler_path="C:\\Program Files\\poppler-26.02.0\\Library\\bin")

    pages = loader.load("data/raw/invoice.pdf")

    print(type(pages))
    print(len(pages))
    print(type(pages[0]))
    for i, page in enumerate(pages):
        print(f"Page {i+1}: {page.size}")
