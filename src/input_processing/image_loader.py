from pathlib import Path
from typing import List 

from PIL import Image

class ImageLoader:
    """
    Load an image file and return it as a list containing one PIL Image.
    """

    def __init__(self):
        pass

    def load(self, image_path: str | Path) -> List[Image.Image]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )
        image = Image.open(image_path)

        return [image]

if __name__ == "__main__":
    loader = ImageLoader()
    pages = loader.load("data/raw/iv01.jpg")

    print(type(pages))
    print(len(pages))
    print(type(pages[0]))
    print(pages[0].size)