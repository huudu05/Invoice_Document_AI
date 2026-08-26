from dataclasses import dataclass, field
from PIL import Image


@dataclass
class LayoutLMInput:
    page_number: int
    image: Image.Image
    words: list[str]
    boxes: list[list[int]]
    word_ids: list[int]

