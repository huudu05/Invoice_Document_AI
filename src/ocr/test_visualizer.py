from src.input_processing.input_manager import InputManager
from src.input_processing.image_preprocessor import ImagePreprocessor

from src.ocr.ocr_processor import OCRProcessor
from src.ocr.visualizer import OCRVisualizer


manager = InputManager()

pages = manager.load("data/raw/invoice.pdf")

pages = ImagePreprocessor().process(pages)

result = OCRProcessor().recognize(pages)

visualizer = OCRVisualizer()

images = visualizer.visualize_document(
    pages,
    result
)

visualizer.save_images(
    images,
    "outputs/visualization"
)

print("Visualization completed.")