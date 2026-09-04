from src.processing.data_normalizer import DataNormalizer


def main():

    print("=" * 60)
    print("TEST DATA NORMALIZER")
    print("=" * 60)

    entities = {
        "company":
            "HOME MASTER HARDWARE& ELECTRICAL",

        "date":
            "22/12/201714:03",

        "address":
            "U13/G BANDARSETIA ALA, "
            "40170 BANDARSETIA ALA, "
            "SELANGOR. TAANOCCEE",

        "total":
            "15.90"
    }

    print("\nRAW ENTITIES")
    print("-" * 60)

    for key, value in entities.items():

        print(
            f"{key:<10}: {value}"
        )

    normalizer = DataNormalizer()

    normalized = normalizer.normalize(
        entities
    )

    print("\nNORMALIZED ENTITIES")
    print("-" * 60)

    for key, value in normalized.items():

        print(
            f"{key:<10}: {value}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()