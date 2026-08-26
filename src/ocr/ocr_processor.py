from typing import List
import torch
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from src.ocr.models import OCRResult
from src.ocr.parser import OCRParser

from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor


class OCRProcessor:
    """
    Extract text from document images using PaddleOCR.
    """

    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en'
        )
        self.parser = OCRParser()

    def recognize(self, pages: List[Image.Image]) -> OCRResult:
        """
        Run OCR on document images.
        """
        raw_results = []
        for page in pages:
            page = np.array(page)
            result = self.ocr.ocr(page, cls=True)
            raw_results.append(result[0])
        return self.parser.parse(raw_results)

if __name__ == "__main__":

    manager = InputManager()
    pages = manager.load("data/raw/invoice.pdf")
    preprocessor = ImagePreprocessor()
    pages = preprocessor.process(pages)
    ocr = OCRProcessor()
    results = ocr.recognize(pages)
    print(type(results)) #<class 'src.ocr.models.OCRResult'>

        