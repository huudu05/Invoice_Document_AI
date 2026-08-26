from datasets import load_dataset
from transformers import LayoutLMv3Processor


MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_LENGTH = 512


def main():

    print("=" * 60)
    print("PREPARE SROIE FOR LAYOUTLMV3")
    print("=" * 60)
    print("\n[1] Loading SROIE dataset...")

    dataset = load_dataset("mp-02/sroie")

    print(dataset)

    train_dataset = dataset["train"]

    print(f"Train samples: {len(train_dataset)}")

    print("\n[2] Loading label information...")

    ner_feature = train_dataset.features["ner_tags"]

    label_names = ner_feature.feature.names

    label2id = {
        label: idx
        for idx, label in enumerate(label_names)
    }

    id2label = {
        idx: label
        for idx, label in enumerate(label_names)
    }

    print("Labels:")

    for idx, label in enumerate(label_names):
        print(f"{idx}: {label}")


    print("\n[3] Loading LayoutLMv3 processor...")

    processor = LayoutLMv3Processor.from_pretrained(
        MODEL_NAME,
        apply_ocr=False
    )

    print("Processor loaded.")

    print("\n[4] Preparing one sample...")

    sample = train_dataset[0]

    image = sample["image"]
    words = sample["words"]
    boxes = sample["bboxes"]
    ner_tags = sample["ner_tags"]

    print(f"Image size: {image.size}")
    print(f"Number of words: {len(words)}")
    print(f"Number of boxes: {len(boxes)}")
    print(f"Number of labels: {len(ner_tags)}")


    labels = []

    for tag in ner_tags:
        labels.append(int(tag))

    print("\n[5] Running LayoutLMv3Processor...")

    encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=labels,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    print("\n[6] Processor output:")

    for key, value in encoding.items():
        print(
            f"{key}: "
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}"
        )

    print("\n[7] Labels:")

    encoded_labels = encoding["labels"]

    print(
        f"labels shape: {tuple(encoded_labels.shape)}"
    )

    print(
        f"labels dtype: {encoded_labels.dtype}"
    )

    print("\n[8] Token / Label inspection:")

    input_ids = encoding["input_ids"][0]
    labels_tensor = encoding["labels"][0]

    tokens = processor.tokenizer.convert_ids_to_tokens(
        input_ids
    )

    for i in range(min(50, len(tokens))):

        token = tokens[i]
        label_id = labels_tensor[i].item()

        if label_id == -100:
            label_name = "IGNORED"
        else:
            label_name = id2label[label_id]

        print(
            f"{i:3d} | "
            f"{token:20s} | "
            f"{label_name}"
        )


    print("\n[9] Encoded label statistics:")

    valid_labels = []

    for label_id in labels_tensor.tolist():

        if label_id != -100:
            valid_labels.append(label_id)

    for label_id, label_name in id2label.items():

        count = valid_labels.count(label_id)

        print(
            f"{label_id}: "
            f"{label_name:15s} "
            f"{count}"
        )

    print("\n[10] Verification:")

    expected_keys = [
        "input_ids",
        "attention_mask",
        "bbox",
        "pixel_values",
        "labels"
    ]

    success = True

    for key in expected_keys:

        if key in encoding:
            print(f"[OK] {key}")
        else:
            print(f"[ERROR] Missing: {key}")
            success = False


    if success:
        print("\nSUCCESS!")
        print(
            "SROIE sample can be converted "
            "to LayoutLMv3 training format."
        )
    else:
        print("\nFAILED!")
        print(
            "Some required LayoutLMv3 inputs are missing."
        )


if __name__ == "__main__":
    main()