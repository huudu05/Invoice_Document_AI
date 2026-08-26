from typing import List

import cv2 
import numpy as np

from PIL import Image

from src.input_processing.image_loader import ImageLoader

class ImagePreprocessor:
    """
    Preprocess document images before OCR.
    """

    def __init__(self, max_side: int = 2000):
        self.max_side = max_side

    def pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        return np.array(image)

    def numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    def convert_rgb(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def resize(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        longest_side = max(width, height)

        if longest_side <= self.max_side:
            return image

        scale = self.max_side / longest_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def process(self, images: List[Image.Image]) -> List[Image.Image]:
        processed_images = []
        for image in images:
            image = self.convert_rgb(image)
            image = self.resize(image)
            processed_images.append(image)
        return processed_images

if __name__ == "__main__":
    loader = ImageLoader()

    pages = loader.load("data/raw/iv01.jpg")

    preprocessor = ImagePreprocessor()

    print(pages[0].mode)

    rgb = preprocessor.convert_rgb(
        pages[0]
    )

    print(rgb.mode)