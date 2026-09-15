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
 
    def enhance_contrast(self, image: Image.Image) -> Image.Image:
        array = self.pil_to_numpy(image)
        lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
 
        lab = cv2.merge((l_channel, a_channel, b_channel))
        array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return self.numpy_to_pil(array)
 
    def denoise(self, image: Image.Image) -> Image.Image:
        array = self.pil_to_numpy(image)
        denoised = cv2.fastNlMeansDenoisingColored(
            array, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
        )
        return self.numpy_to_pil(denoised)
 
    def deskew(self, image: Image.Image) -> Image.Image:

        array = self.pil_to_numpy(image)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
 
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 50:
            return image
 
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
 
        if abs(angle) < 0.5 or abs(angle) > 15:
            return image
 
        height, width = array.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            array, rotation_matrix, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return self.numpy_to_pil(rotated)
 
    def process(
        self,
        images: List[Image.Image],
        enhance: bool = False,
    ) -> List[Image.Image]:

        processed_images = []
        for image in images:
            image = self.convert_rgb(image)
            image = self.resize(image)
            if enhance:
                image = self.deskew(image)
                image = self.denoise(image)
                image = self.enhance_contrast(image)
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
 