from src.training.sroie_dataset import SROIEDataset


def main():
    print("=" * 60)
    print("TEST SROIE DATASET")
    print("=" * 60)

    dataset = SROIEDataset(split="train")

    print(f"Number of samples: {len(dataset)}")
    print(f"Labels: {dataset.label_names}")

    sample = dataset[0]

    print("\nSample keys:")
    print(sample.keys())

    print("\nTensor shapes:")

    for key, value in sample.items():
        print(f"{key:20} {tuple(value.shape)} {value.dtype}")

    print("\nLabels:")
    print(sample["labels"][:50])

    print("\nBounding boxes:")
    print(sample["bbox"][:10])

    print("\nInput IDs:")
    print(sample["input_ids"][:20])


if __name__ == "__main__":
    main()