import torch
from torch.utils.data import DataLoader, Subset

from src.training.sroie_dataset import SROIEDataset
from src.training.layoutlm_trainer import LayoutLMTrainer


def main():

    print("=" * 60)
    print("TEST LAYOUTLMV3 TRAINER + METRICS")
    print("=" * 60)

    # 1. Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # 2. Dataset
    print("\n[1] Loading dataset...")

    dataset = SROIEDataset()

    print(
        f"Full dataset size: "
        f"{len(dataset)}"
    )

    # 3. Small subset
    print("\n[2] Creating test subset...")

    subset = Subset(
        dataset,
        list(range(10))
    )

    print(
        f"Subset size: {len(subset)}"
    )

    # 4. DataLoader
    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False
    )

    # 5. Trainer
    print("\n[3] Creating trainer...")

    trainer = LayoutLMTrainer(
        num_labels=5,
        learning_rate=5e-5,
        device=device
    )

    print("Trainer created.")

    # 6. Evaluate before training
    print("\n[4] Initial evaluation...")

    metrics = trainer.evaluate(
        loader
    )

    trainer.print_metrics(
        metrics
    )

    # 7. Train for one epoch
    print("\n[5] Training one epoch...")

    total_loss = 0.0

    for step, batch in enumerate(loader):

        loss = trainer.train_step(
            batch
        )

        total_loss += loss

        print(
            f"Step {step + 1:02d} "
            f"| Loss: {loss:.6f}"
        )

    avg_loss = (
        total_loss / len(loader)
    )

    print(
        f"\nAverage training loss: "
        f"{avg_loss:.6f}"
    )

    # 8. Evaluate after training
    print("\n[6] Evaluation after training...")

    metrics = trainer.evaluate(
        loader
    )

    trainer.print_metrics(
        metrics
    )

    # 9. Verification
    print("\n[7] Verification...")

    assert metrics["loss"] >= 0

    assert 0.0 <= metrics["accuracy"] <= 1.0

    assert 0.0 <= metrics["precision"] <= 1.0

    assert 0.0 <= metrics["recall"] <= 1.0

    assert 0.0 <= metrics["f1"] <= 1.0

    assert len(
        metrics["per_label"]
    ) == 5

    print("[OK] Loss")
    print("[OK] Accuracy")
    print("[OK] Precision")
    print("[OK] Recall")
    print("[OK] F1")
    print("[OK] Per-label metrics")

    print("\n" + "=" * 60)
    print("TRAINER + METRIC TEST SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    main()