import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Tuple


import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

from src.training.sroie_dataset import SROIEDataset

DEFAULT_MODEL_DIR = Path(
    "models/layoutlmv3/best_model"
)

DEFAULT_OUTPUT_DIR = Path(
    "models/layoutlmv3/evaluation/test"
)

DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 0

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fine-tuned LayoutLMv3 "
            "on the SROIE test set."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=(
            "Path to the fine-tuned LayoutLMv3 model."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory where evaluation results "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader workers.",
    )

    return parser.parse_args()

def get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(
    model_dir: Path,
    device: torch.device,
) -> LayoutLMv3ForTokenClassification:

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}"
        )

    print()
    print(
        f"Loading model from: {model_dir}"
    )

    model = (
        LayoutLMv3ForTokenClassification
        .from_pretrained(model_dir)
    )

    model.to(device)
    model.eval()

    print("[OK] Model loaded.")

    return model


def load_test_dataset() -> SROIEDataset:

    print()
    print(
        "Loading SROIE test dataset..."
    )

    dataset = SROIEDataset(
        split="test",
        model_name="microsoft/layoutlmv3-base",
    )

    print(
        f"[OK] Test samples: {len(dataset)}"
    )

    return dataset

def collect_predictions(
    model: LayoutLMv3ForTokenClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[
    List[int],
    List[int],
    float,
]:

    all_predictions: List[int] = []
    all_labels: List[int] = []

    total_loss = 0.0
    num_batches = 0

    print()
    print(
        "Running inference..."
    )

    with torch.no_grad():

        for batch_index, batch in enumerate(
            dataloader,
            start=1,
        ):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )

            total_loss += outputs.loss.item()
            num_batches += 1

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            labels = batch["labels"]

            predictions = predictions.cpu()
            labels = labels.cpu()

            for prediction, label in zip(
                predictions,
                labels,
            ):

                for pred_value, label_value in zip(
                    prediction,
                    label,
                ):

                    label_id = label_value.item()

                    if label_id == -100:
                        continue

                    all_predictions.append(
                        pred_value.item()
                    )

                    all_labels.append(
                        label_id
                    )

            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == len(dataloader)
            ):

                print(
                    f"Batch "
                    f"{batch_index:03d}/"
                    f"{len(dataloader):03d}"
                )

    if not all_labels:
        raise RuntimeError(
            "No valid labels were found."
        )

    average_loss = (
        total_loss / num_batches
    )

    return (
        all_labels,
        all_predictions,
        average_loss,
    )

def calculate_metrics(
    labels: List[int],
    predictions: List[int],
    label_names: List[str],
    loss: float,
) -> Dict[str, Any]:

    label_ids = list(
        range(len(label_names))
    )
    accuracy = accuracy_score(
        labels,
        predictions,
    )

    # Macro metrics
    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )

    # Weighted metrics
    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )

  
    # Per-label metrics
    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        zero_division=0,
    )

    per_label = {}

    for label_id, label_name in enumerate(
        label_names
    ):

        per_label[label_name] = {
            "precision": float(
                precision[label_id]
            ),
            "recall": float(
                recall[label_id]
            ),
            "f1": float(
                f1[label_id]
            ),
            "support": int(
                support[label_id]
            ),
        }


    # Entity metrics
    entity_label_ids = [
        label_id
        for label_id, label_name
        in enumerate(label_names)
        if label_name != "O"
    ]

    (
        entity_precision,
        entity_recall,
        entity_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=entity_label_ids,
        average="macro",
        zero_division=0,
    )

    # Classification report
    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )


    # Result
    return {
        "loss": float(loss),

        "accuracy": float(accuracy),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "entity_precision": float(
            entity_precision
        ),

        "entity_recall": float(
            entity_recall
        ),

        "entity_f1": float(
            entity_f1
        ),

        "per_label": per_label,

        "classification_report": report,

        "num_tokens": len(labels),
    }


# CONFUSION MATRIX
def save_confusion_matrix(
    labels: List[int],
    predictions: List[int],
    label_names: List[str],
    output_path: Path,
    normalize: bool = False,
) -> np.ndarray:

    label_ids = list(
        range(len(label_names))
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=label_ids,
    )

    if normalize:

        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        cm_display = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=row_sums != 0,
        )

    else:

        cm_display = cm

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        cm_display,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(
            len(label_names)
        ),
        yticks=np.arange(
            len(label_names)
        ),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="Actual",
        xlabel="Predicted",
        title=(
            "Normalized Confusion Matrix"
            if normalize
            else "Confusion Matrix"
        ),
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        cm_display.max() / 2.0
        if cm_display.size
        else 0
    )

    for row in range(
        cm_display.shape[0]
    ):

        for col in range(
            cm_display.shape[1]
        ):

            if normalize:

                text = (
                    f"{cm_display[row, col]:.2f}"
                )

            else:

                text = str(
                    cm_display[row, col]
                )

            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm_display[row, col]
                    > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return cm


def save_json(
    data: Dict[str, Any],
    path: Path,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def print_results(
    metrics: Dict[str, Any],
) -> None:

    print()
    print("=" * 75)
    print("SROIE TEST SET EVALUATION")
    print("=" * 75)
    print(
        f"Loss              : "
        f"{metrics['loss']:.6f}"
    )
    print(
        f"Accuracy          : "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Macro Precision   : "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall      : "
        f"{metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1          : "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted Precision: "
        f"{metrics['weighted_precision']:.4f}"
    )
    print(
        f"Weighted Recall   : "
        f"{metrics['weighted_recall']:.4f}"
    )
    print(
        f"Weighted F1       : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print()
    print(
        f"Entity Precision  : "
        f"{metrics['entity_precision']:.4f}"
    )
    print(
        f"Entity Recall     : "
        f"{metrics['entity_recall']:.4f}"
    )
    print(
        f"Entity F1         : "
        f"{metrics['entity_f1']:.4f}"
    )
    print(
        f"Valid Tokens      : "
        f"{metrics['num_tokens']}"
    )

    print()
    print("Per-label metrics:")
    print(
        f"{'Label':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print("-" * 63)
    for label_name, values in (
        metrics["per_label"].items()
    ):

        print(
            f"{label_name:<15}"
            f"{values['precision']:>12.4f}"
            f"{values['recall']:>12.4f}"
            f"{values['f1']:>12.4f}"
            f"{values['support']:>12}"
        )

    print(
        "=" * 75
    )



def main():

    args = parse_args()

    model_dir = Path(
        args.model_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than 0."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers cannot be negative."
        )

    device = get_device()

    print()
    print("=" * 75)
    print("LAYOUTLMV3 FINAL EVALUATION")
    print("=" * 75)
    print(
        f"Model directory : {model_dir}"
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"Batch size      : {args.batch_size}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        "=" * 75
    )

    # Dataset
    dataset = load_test_dataset()

    label_names = dataset.label_names

    print()
    print(
        "Labels:"
    )

    for label_id, label_name in enumerate(
        label_names
    ):

        print(
            f"  {label_id}: {label_name}"
        )


    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print(
        f"Test batches: {len(dataloader)}"
    )


    # Processor check
    processor_path = (
        model_dir
    )

    if processor_path.exists():
        try:
            processor = (
                LayoutLMv3Processor
                .from_pretrained(
                    processor_path,
                    apply_ocr=False,
                )
            )
            print("[OK] Saved processor found.")

        except Exception as exc:
            processor = None
            print(
                "[WARNING] Could not load "
                f"saved processor: {exc}"
            )

    else:
        processor = None


    # Model
    model = load_model(
        model_dir,
        device,
    )

    # Inference

    (
        labels,
        predictions,
        loss,
    ) = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    # Metrics
    metrics = calculate_metrics(
        labels=labels,
        predictions=predictions,
        label_names=label_names,
        loss=loss,
    )

    # Print
    print_results(
        metrics
    )

    # Save metrics
    metrics_to_save = {
        key: value
        for key, value in metrics.items()
        if key != "classification_report"
    }

    save_json(
        metrics_to_save,
        output_dir / "test_metrics.json",
    )

    save_json(
        metrics["classification_report"],
        output_dir / "classification_report.json",
    )

    # Confusion matrix
    cm = save_confusion_matrix(
        labels=labels,
        predictions=predictions,
        label_names=label_names,
        output_path=(
            output_dir
            / "confusion_matrix.png"
        ),
        normalize=False,
    )

    normalized_cm = save_confusion_matrix(
        labels=labels,
        predictions=predictions,
        label_names=label_names,
        output_path=(
            output_dir
            / "confusion_matrix_normalized.png"
        ),
        normalize=True,
    )

    save_json(
        {
            "labels": label_names,
            "matrix": cm.tolist(),
        },
        output_dir / "confusion_matrix.json",
    )

    save_json(
        {
            "labels": label_names,
            "matrix": normalized_cm.tolist(),
        },
        output_dir
        / "confusion_matrix_normalized.json",
    )

    # Save predictions
    save_json(
        {
            "labels": labels,
            "predictions": predictions,
            "num_tokens": len(labels),
        },
        output_dir / "predictions.json",
    )

    # Save evaluation metadata
    metadata = {
        "model_dir": str(model_dir),
        "dataset": "mp-02/sroie",
        "split": "test",
        "num_samples": len(dataset),
        "num_valid_tokens": len(labels),
        "device": str(device),
        "batch_size": args.batch_size,
        "label_names": label_names,
    }

    save_json(
        metadata,
        output_dir / "evaluation_config.json",
    )

    # Finished
    print()
    print(
        "=" * 75
    )
    print(
        "EVALUATION COMPLETED"
    )
    print(
        "=" * 75
    )
    print(
        f"Metrics              : "
        f"{output_dir / 'test_metrics.json'}"
    )
    print(
        f"Classification report: "
        f"{output_dir / 'classification_report.json'}"
    )

    print(
        f"Confusion matrix     : "
        f"{output_dir / 'confusion_matrix.png'}"
    )

    print(
        f"Normalized matrix    : "
        f"{output_dir / 'confusion_matrix_normalized.png'}"
    )

    print(
        f"Predictions          : "
        f"{output_dir / 'predictions.json'}"
    )

    print(
        "=" * 75
    )

if __name__ == "__main__":
    main()