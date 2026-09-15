from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from .models import OCRPage
from .models import OCRResult

class OCRVisualizer:


    ENTITY_COLOR_MAP = {
        "company": "#8E44AD",  
        "date": "#E67E22",      
        "address": "#2980B9",   
        "total": "#C0392B",     
    }

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

    @staticmethod
    def _label_to_entity(label: str) -> Optional[str]:
        if not label:
            return None
 
        label = label.upper()
        if label == "O":
            return None
 
        entity = label.split("-", 1)[1] if "-" in label else label
        return entity.lower()
 
    def draw_entities(
            self,
            image: Image.Image,
            page: OCRPage,
            word_labels: List[str],
            color_map: Optional[Dict[str, str]] = None,
            width: int = 3,
            show_legend: bool = True,
    ):
 
        if len(word_labels) != len(page.words):
            raise ValueError(
                "word_labels phải có cùng độ dài với page.words "
                f"({len(word_labels)} != {len(page.words)})."
            )
 
        color_map = color_map or self.ENTITY_COLOR_MAP
 
        image = image.copy()
        draw = ImageDraw.Draw(image)
 
        for word, label in zip(page.words, word_labels):
            entity = self._label_to_entity(label)
            color = color_map.get(entity) if entity else None
            if color is None:
                continue
 
            polygon = [tuple(pt) for pt in word.bbox]
            polygon.append(tuple(word.bbox[0]))
            draw.line(polygon, fill=color, width=width)
 
        if show_legend:
            image = self._draw_legend(image, color_map)
 
        return image
 
    def _draw_legend(self, image: Image.Image, color_map: Dict[str, str]):
        image = image.copy()
        draw = ImageDraw.Draw(image)
 
        x, y = 10, 10
        box_size = 16
        gap = 6
 
        for entity, color in color_map.items():
            draw.rectangle(
                [x, y, x + box_size, y + box_size],
                fill=color,
                outline="black",
            )
            draw.text(
                (x + box_size + gap, y - 2),
                entity.capitalize(),
                fill="black",
                font=self.font,
            )
            y += box_size + gap + 6
 
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
        