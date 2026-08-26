from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from .models import OCRPage
from .models import OCRResult

class OCRVisualizer:
    def __init__(self, font_size=18):
        try:
            self.font = ImageFont.truetype("arial.ttf", font_size)
        except:
            self.font = ImageFont.load_default()

    def draw_boxes(
            self,
            image: Image.Image,
            page: OCRPage,
            color='red',
            width=2,
    ):

        image = image.copy()
        draw = ImageDraw.Draw(image)
        for word in page.words:
            polygon = [tuple(pt) for pt in word.bbox]
            polygon.append(tuple(word.bbox[0]))
            draw.line(polygon, fill=color, width=width)
        return image

    def draw_text(
            self, 
            image: Image.Image, 
            page: OCRPage,
            color='blue'
    ):
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for word in page.words:
            x = word.bbox[0][0]
            y = word.bbox[0][1] - 18

            draw.text((x,y), word.text, fill=color, font=self.font)

        return image

    def draw_ids(
            self, 
            image: Image.Image,
            page: OCRPage,
            color='green'
    ):
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for word in page.words:
            x = word.bbox[0][0]
            y = word.bbox[0][1]

            draw.text((x,y), str(word.id), fill=color, font=self.font)

        return image

    def draw_confidence(
            self,
            image: Image.Image,
            page: OCRPage,
            color='orange'
    ):
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for word in page.words:
            x = word.bbox[3][0]
            y = word.bbox[3][1]

            draw.text((x,y), f"{word.confidence:.2f}", fill=color, font=self.font)

        return image

    def visualize_page(
            self,
            image: Image.Image,
            page: OCRPage
    ):
        image = self.draw_boxes(image, page)
        image = self.draw_ids(image, page)

        return image

    def visualize_document(
            self, 
            images: List[Image.Image],
            result: OCRResult
    ):
        outputs = []
        for image, page in zip(images, result.pages):
            outputs.append(self.visualize_page(image, page))

        return outputs

    def save_images(self, images: List[Image.Image], output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            image.save(output_dir / f"page_{i+1}.png")
        