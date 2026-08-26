from .models import OCRResult
from .models import OCRPage
from .models import OCRWord


class OCRParser:

    def parse(self, raw_results):
        pages = []

        for page_idx, page in enumerate(raw_results):
            words = []

            for word_index, item in enumerate(page):
                bbox = item[0]
                text = item[1][0]
                confidence = item[1][1]
                words.append(
                    OCRWord(
                        id= word_index,
                        text=text,
                        confidence=confidence,
                        bbox=bbox
                    )
                )

            pages.append(
                OCRPage(
                    page_number=page_idx + 1,
                    words=words
                )
            )

        return OCRResult(pages=pages)