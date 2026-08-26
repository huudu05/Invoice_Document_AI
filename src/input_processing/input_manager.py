from pathlib import Path
from typing import List

from PIL import Image

from src.input_processing.pdf_loader import PdfLoader
from src.input_processing.image_loader import ImageLoader
from src.input_processing.file_validator import FileValidator

class InputManager:
    """
    Manage document loading by selecting the appropriate loader
    based on the input file type.
    """

    def __init__(self):
        self.validator = FileValidator()
        self.pdf_loader = PdfLoader(poppler_path="C:\\Program Files\\poppler-26.02.0\\Library\\bin")
        self.image_loader = ImageLoader()

    def load(self, file_path: str | Path) -> List[Image.Image]:
        file_path = Path(file_path)

        result = self.validator.validate(file_path)
        if not result.is_valid:
            raise ValueError(result.message)
        
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return self.pdf_loader.load(file_path)
        elif suffix in ['.jpg', '.jpeg', '.png']:
            return self.image_loader.load(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

if __name__ == "__main__":
    manager = InputManager()
    pages = manager.load("data/raw/invoice.pdf")
    print(len(pages))

    pages = manager.load("data/raw/iv01.jpg")
    print(len(pages))
    
        