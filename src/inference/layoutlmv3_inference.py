from typing import List, Dict

import torch
from PIL import Image

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

from src.inference.entity_extractor import EntityExtractor


class LayoutLMv3Inference:

    def __init__(
        self,
        model_path: str = "models/layoutlmv3_sroie/best",
        device: str = None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print("=" * 60)
        print("LOADING FINE-TUNED LAYOUTLMV3")
        print("=" * 60)

        print(f"Device: {self.device}")
        print(f"Model path: {model_path}")


        # PROCESSOR
        print("\nLoading processor...")

        self.processor = (
            LayoutLMv3Processor.from_pretrained(
                model_path,
                apply_ocr=False,
            )
        )


        # MODEL
        print("Loading fine-tuned model...")

        self.model = (
            LayoutLMv3ForTokenClassification
            .from_pretrained(model_path)
        )

        self.model.to(self.device)
        self.model.eval()


        # LABEL MAPPING
        self.id2label = self.model.config.id2label
        print(f"Labels: {self.id2label}")


        # ENTITY EXTRACTOR
        self.entity_extractor = EntityExtractor(
            self.id2label
        )

        print("\nLayoutLMv3 loaded successfully.")


    # NORMALIZE BBOX
    @staticmethod
    def normalize_bbox(
        bbox,
        image_width: int,
        image_height: int,
    ):
        """
        Convert pixel coordinates to LayoutLMv3
        normalized coordinates [0, 1000].
        """

        x0, y0, x1, y1 = bbox

        x0 = int(1000 * x0 / image_width)
        y0 = int(1000 * y0 / image_height)

        x1 = int(1000 * x1 / image_width)
        y1 = int(1000 * y1 / image_height)

        return [
            max(0, min(1000, x0)),
            max(0, min(1000, y0)),
            max(0, min(1000, x1)),
            max(0, min(1000, y1)),
        ]


    # PREDICT
    def predict(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[List[int]],
    ) -> Dict[str, str]:


        # VALIDATION
        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        if len(words) != len(boxes):
            raise ValueError(
                "Number of words must equal "
                "number of bounding boxes."
            )

        if len(words) == 0:
            raise ValueError(
                "No OCR words were provided."
            )


        # IMAGE SIZE
        image_width, image_height = image.size

        print(
            f"\nImage size: "
            f"{image_width} x {image_height}"
        )

        print(
            f"OCR words: {len(words)}"
        )


        # NORMALIZE BBOX
        normalized_boxes = [
            self.normalize_bbox(
                box,
                image_width,
                image_height,
            )
            for box in boxes
        ]


        # PROCESSOR
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # MOVE TO DEVICE
        encoding = encoding.to(
            self.device
        )


        # MODEL INFERENCE
        with torch.no_grad():

            outputs = self.model(
                **encoding
            )


        # TOKEN PREDICTIONS
        predictions = (
            outputs.logits
            .argmax(dim=-1)
            .squeeze(0)
            .cpu()
            .tolist()
        )

        # WORD IDS
        word_ids = encoding.word_ids(
            batch_index=0
        )


        # TOKEN → OCR WORD
        word_predictions = [
            None
        ] * len(words)

        for prediction, word_id in zip(
            predictions,
            word_ids,
        ):

            if word_id is None:
                continue

            if word_id >= len(words):
                continue

            if word_predictions[word_id] is None:

                word_predictions[word_id] = (
                    prediction
                )

        # O LABEL
        o_label_id = next(
            (
                label_id
                for label_id, label
                in self.id2label.items()
                if label == "O"
            ),
            4,
        )

        # FILL MISSING WORD PREDICTIONS
        word_predictions = [
            prediction
            if prediction is not None
            else o_label_id
            for prediction in word_predictions
        ]

        # DEBUG
        print("\nLAYOUTLMV3 PREDICTIONS")
        print("-" * 80)

        for index, (
            word,
            prediction,
        ) in enumerate(
            zip(words, word_predictions)
        ):

            label = self.id2label.get(
                prediction,
                "O",
            )

            print(
                f"{index:03d} | "
                f"{word:<35} | "
                f"{label}"
            )

        # ENTITY EXTRACTION
        entities = self.entity_extractor.extract(
            words,
            word_predictions,
        )


        # RESULT
        print("\nEXTRACTED ENTITIES")
        print("-" * 80)

        for key, value in entities.items():

            print(
                f"{key:<15}: {value}"
            )

        return entities