from collections import Counter

from datasets import load_dataset


def main():

    print("=" * 60)
    print("ANALYZE SROIE LABELS")
    print("=" * 60)

    dataset = load_dataset("mp-02/sroie")

    label_feature = dataset["train"].features["ner_tags"].feature

    id2label = {
        idx: label
        for idx, label in enumerate(label_feature.names)
    }

    print("\nLabel mapping:")

    for idx, label in id2label.items():
        print(f"{idx}: {label}")
    # Label mapping:
    # 0: S-COMPANY
    # 1: S-DATE
    # 2: S-ADDRESS
    # 3: S-TOTAL
    # 4: O


    for split in ["train", "test"]:

        counter = Counter()

        for sample in dataset[split]:

            for label_id in sample["ner_tags"]:
                counter[label_id] += 1

        print("\n" + "-" * 60)
        print(f"{split.upper()} LABEL DISTRIBUTION")
        print("-" * 60)

        total = sum(counter.values())

        for label_id in sorted(counter):

            label_name = id2label[label_id]
            count = counter[label_id]
            percentage = count / total * 100

            print(
                f"{label_id}: "
                f"{label_name:<12} "
                f"{count:>7} "
                f"({percentage:>6.2f}%)"
            )

        print(f"\nTotal tokens: {total}")


if __name__ == "__main__":
    main()