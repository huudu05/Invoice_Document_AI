from PIL import Image
from src.ocr.models import OCRResult
from src.inference.input_models import LayoutLMInput

class LayoutLMInputBuilder:
    
    def quad_to_box(self, quad):
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]

        return [
            int(min(xs)),
            int(min(ys)),
            int(max(xs)),
            int(max(ys))
        ]

    def _normalize_box(self, box, width, height):
        xmin, ymin, xmax, ymax = box

        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))
        
        return [
            int(1000* xmin / width),
            int(1000 * ymin / height),
            int(1000 * xmax / width),
            int(1000 * ymax / height)
        ]

    def build(self, images: list[Image.Image], ocr_result:OCRResult):
        outputs = []
        for image, page in zip(images, ocr_result.pages):
            words = []
            boxes = []
            word_ids = []

            width, height = image.size

            for word in page.words:
                words.append(word.text)
                word_ids.append(word.id)
                box = self.quad_to_box(word.bbox)
                box = self._normalize_box(box, width, height)
                boxes.append(box)

            outputs.append(LayoutLMInput(
                page_number=page.page_number,
                image=image,
                words=words,
                boxes=boxes,
                word_ids=word_ids
            ))
        return outputs