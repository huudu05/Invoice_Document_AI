from dataclasses import dataclass, field

@dataclass
class OCRWord:
    id : int
    text: str
    confidence: float
    bbox: list[list[int]]

@dataclass
class OCRPage:
    page_number : int
    words: list[OCRWord] = field(default_factory=list)

@dataclass
class OCRResult:
    pages: list[OCRPage] = field(default_factory=list)
    