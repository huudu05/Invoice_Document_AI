from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor

from src.ocr.ocr_processor import OCRProcessor
from src.ocr.exporter import OCRExporter


pages = InputManager().load("data/raw/invoice.pdf")

pages = ImagePreprocessor().process(pages)

result = OCRProcessor().recognize(pages)

exporter = OCRExporter()

exporter.export_json(
    result,
    "outputs/result.json"
)

exporter.export_txt(
    result,
    "outputs/result.txt"
)

exporter.export_csv(
    result,
    "outputs/result.csv"
)

print(exporter.to_json(result)[:500])

print("Export completed.")