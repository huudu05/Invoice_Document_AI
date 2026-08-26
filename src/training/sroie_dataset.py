from typing import Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import LayoutLMv3Processor


class SROIEDataset(Dataset):
    """
    SROIE dataset prepared for LayoutLMv3 token classification.

    The dataset provides:
    - invoice image
    - OCR words
    - bounding boxes
    - token-level NER labels

    The LayoutLMv3 processor converts these into:
    - input_ids
    - attention_mask
    - bbox
    - labels
    - pixel_values
    """

    def __init__(
        self,
        split: str = "train",
        model_name: str = "microsoft/layoutlmv3-base",
        max_length: int = 512,
    ):

        self.dataset = load_dataset(
            "mp-02/sroie",
            split=split,
        )

        self.processor = (
            LayoutLMv3Processor.from_pretrained(
                model_name,
                apply_ocr=False,
            )
        )

        self.max_length = max_length


        # Label mapping
        ner_feature = (
            self.dataset.features["ner_tags"].feature
        )

        self.label_names = ner_feature.names

        self.label2id = {
            label: idx
            for idx, label in enumerate(
                self.label_names
            )
        }

        self.id2label = {
            idx: label
            for idx, label in enumerate(
                self.label_names
            )
        }

    def __len__(self):
        return len(self.dataset)


    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:

        item = self.dataset[idx]

        image = item["image"]
        words = item["words"]
        boxes = item["bboxes"]
        word_labels = item["ner_tags"]

        # LayoutLMv3 preprocessing
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension
        encoding = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        return encoding