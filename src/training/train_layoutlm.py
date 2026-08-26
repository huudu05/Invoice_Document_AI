import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.training.sroie_dataset import SROIEDataset
from src.training.layoutlm_trainer import LayoutLMTrainer


SEED = 42

MODEL_NAME = "microsoft/layoutlmv3-base"

VAL_RATIO = 0.10

EPOCHS = 10
BATCH_SIZE = 2

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CLIP_MAX_NORM = 1.0

NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 1e-4

OUTPUT_DIR = Path(
    "models/layoutlmv3"
)


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train LayoutLMv3 on SROIE "
            "for invoice information extraction."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            f"Number of epochs. "
            f"Default: {EPOCHS}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            f"Batch size. "
            f"Default: {BATCH_SIZE}"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=(
            f"Learning rate. "
            f"Default: {LEARNING_RATE}"
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=(
            f"Weight decay. "
            f"Default: {WEIGHT_DECAY}"
        ),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=WARMUP_RATIO,
        help=(
            f"Warmup ratio. "
            f"Default: {WARMUP_RATIO}"
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=(
            "Number of gradient accumulation steps."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=GRADIENT_CLIP_MAX_NORM,
        help=(
            "Maximum gradient norm. "
            f"Default: {GRADIENT_CLIP_MAX_NORM}"
        ),
    )


    parser.add_argument(
        "--train-samples",
        type=str,
        default="100",
        help=(
            "Maximum number of training samples. "
            "Use 'none' for all samples."
        ),
    )
    parser.add_argument(
        "--val-samples",
        type=str,
        default="30",
        help=(
            "Maximum number of validation samples. "
            "Use 'none' for all validation samples."
        ),
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=(
            f"Validation ratio. "
            f"Default: {VAL_RATIO}"
        ),
    )


    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=(
            f"DataLoader workers. "
            f"Default: {NUM_WORKERS}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            f"Random seed. "
            f"Default: {SEED}"
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Training output directory.",
    )


    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume from."
        ),
    )

    return parser.parse_args()


def parse_optional_int(
    value: str,
) -> Optional[int]:

    if value.lower() == "none":
        return None

    result = int(value)

    if result <= 0:

        raise ValueError(
            "Sample limit must be positive "
            "or 'none'."
        )

    return result


def create_directories(
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Create experiment directories.
    """

    best_model_dir = (
        output_dir / "best_model"
    )

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    history_file = (
        output_dir / "training_history.json"
    )

    config_file = (
        output_dir / "training_config.json"
    )

    best_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "output_dir": output_dir,
        "best_model_dir": best_model_dir,
        "checkpoint_dir": checkpoint_dir,
        "history_file": history_file,
        "config_file": config_file,
    }



def create_train_val_split(
    dataset: SROIEDataset,
    val_ratio: float,
    train_samples: Optional[int],
    val_samples: Optional[int],
    seed: int,
) -> Tuple[Subset, Subset]:

    if not 0 < val_ratio < 1:

        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    total_size = len(dataset)

    if total_size < 2:

        raise RuntimeError(
            "Dataset must contain at least 2 samples."
        )

    indices = list(
        range(total_size)
    )

    rng = random.Random(seed)

    rng.shuffle(indices)

    val_size = max(
        1,
        int(total_size * val_ratio),
    )

    val_indices = indices[
        :val_size
    ]

    train_indices = indices[
        val_size:
    ]

    if train_samples is not None:

        train_indices = train_indices[
            :train_samples
        ]

    if val_samples is not None:

        val_indices = val_indices[
            :val_samples
        ]

    if not train_indices:

        raise RuntimeError(
            "Training subset is empty."
        )

    if not val_indices:

        raise RuntimeError(
            "Validation subset is empty."
        )

    return (
        Subset(
            dataset,
            train_indices,
        ),
        Subset(
            dataset,
            val_indices,
        ),
    )


def get_rng_state() -> Dict[str, Any]:

    state = {
        "python": random.getstate(),

        "numpy": np.random.get_state(),

        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():

        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: Dict[str, Any],
) -> None:

    if not state:
        return

    if "python" in state:

        random.setstate(
            state["python"]
        )

    if "numpy" in state:

        np.random.set_state(
            state["numpy"]
        )

    if "torch" in state:

        torch.set_rng_state(
            state["torch"]
        )

    if (
        "cuda" in state
        and torch.cuda.is_available()
    ):

        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


def save_checkpoint(
    trainer: LayoutLMTrainer,
    epoch: int,
    metrics: Dict[str, Any],
    best_entity_f1: float,
    epochs_without_improvement: int,
    history: list,
    path: Path,
    args: argparse.Namespace,
) -> None:

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            trainer.model.state_dict(),

        "optimizer_state_dict":
            trainer.optimizer.state_dict(),

        "scheduler_state_dict":
            (
                trainer.scheduler.state_dict()
                if trainer.scheduler is not None
                else None
            ),

        "metrics": metrics,

        "best_entity_f1":
            best_entity_f1,

        "epochs_without_improvement":
            epochs_without_improvement,

        "history":
            history,

        "model_name":
            trainer.model_name,

        "num_labels":
            trainer.num_labels,

        "id2label":
            trainer.id2label,

        "label2id":
            trainer.label2id,

        "learning_rate":
            trainer.learning_rate,

        "weight_decay":
            trainer.weight_decay,

        "max_grad_norm":
            trainer.max_grad_norm,

        "rng_state":
            get_rng_state(),

        "args":
            vars(args),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    trainer: LayoutLMTrainer,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """
    Load complete checkpoint.
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=trainer.device,
        weights_only=False,
    )

    checkpoint_id2label = {
        int(key): value
        for key, value
        in checkpoint["id2label"].items()
    }

    if checkpoint_id2label != trainer.id2label:

        raise ValueError(
            "Checkpoint label mapping does not "
            "match the current dataset."
        )


    # Model


    trainer.model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    # Optimizer


    trainer.optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    # Scheduler


    if (
        trainer.scheduler is not None
        and checkpoint.get(
            "scheduler_state_dict"
        ) is not None
    ):

        trainer.scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )


    # RNG


    restore_rng_state(
        checkpoint.get(
            "rng_state",
            {},
        )
    )

    print(
        f"[OK] Checkpoint loaded: "
        f"{checkpoint_path}"
    )

    return checkpoint


# ============================================================
# HISTORY
# ============================================================

def save_history(
    history: list,
    history_file: Path,
) -> None:
    """
    Save training history.
    """

    with open(
        history_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# CONFIG
# ============================================================

def save_training_config(
    args: argparse.Namespace,
    dataset: SROIEDataset,
    output_dir: Path,
    train_size: int,
    val_size: int,
    device: torch.device,
) -> None:
    """
    Save exact experiment configuration.
    """

    config = {
        "seed": args.seed,

        "model_name":
            MODEL_NAME,

        "epochs":
            args.epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "warmup_ratio":
            args.warmup_ratio,

        "gradient_accumulation_steps":
            args.gradient_accumulation_steps,

        "max_grad_norm":
            args.max_grad_norm,

        "val_ratio":
            args.val_ratio,

        "train_samples":
            args.train_samples,

        "val_samples":
            args.val_samples,

        "num_workers":
            args.num_workers,

        "num_labels":
            len(dataset.label_names),

        "label_names":
            dataset.label_names,

        "label2id":
            dataset.label2id,

        "id2label":
            dataset.id2label,

        "full_dataset_size":
            len(dataset),

        "actual_train_size":
            train_size,

        "actual_val_size":
            val_size,

        "device":
            str(device),

        "output_dir":
            str(output_dir),
    }

    config_file = (
        output_dir /
        "training_config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# METRICS PRINT
# ============================================================

def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    metrics: Dict[str, Any],
    epoch_time: float,
) -> None:

    print()
    print("=" * 75)

    print(
        f"Epoch {epoch}/{total_epochs}"
    )

    print("=" * 75)

    print(
        f"Train Loss        : "
        f"{train_loss:.6f}"
    )

    print(
        f"Validation Loss   : "
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

    print()

    print(
        f"O F1              : "
        f"{metrics['o_f1']:.4f}"
    )

    print(
        f"Learning Rate     : "
        f"{metrics['learning_rate']:.8f}"
    )

    print(
        f"Valid Tokens      : "
        f"{metrics['num_tokens']}"
    )

    print(
        f"Epoch Time        : "
        f"{epoch_time:.2f} sec"
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

    print("=" * 75)


# ============================================================
# BEST MODEL METADATA
# ============================================================

def save_best_model_metadata(
    best_model_dir: Path,
    epoch: int,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """
    Save metadata for best model.
    """

    metadata = {
        "epoch": epoch,

        "entity_f1":
            float(
                metrics["entity_f1"]
            ),

        "metrics": metrics,

        "training_arguments":
            vars(args),
    }

    metadata_path = (
        best_model_dir /
        "metrics.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# SAVE PROCESSOR
# ============================================================

def save_processor(
    dataset: SROIEDataset,
    best_model_dir: Path,
) -> None:
    """
    Save processor together with model.
    """

    dataset.processor.save_pretrained(
        best_model_dir
    )

    print(
        "[OK] Processor saved."
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    trainer: LayoutLMTrainer,
    dataset: SROIEDataset,
    best_model_dir: Path,
    epoch: int,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> None:

    # Model
    trainer.model.save_pretrained(
        best_model_dir
    )

    # Processor
    save_processor(
        dataset,
        best_model_dir,
    )

    # Metadata
    save_best_model_metadata(
        best_model_dir,
        epoch,
        metrics,
        args,
    )

    print(
        f"[OK] Best model saved to: "
        f"{best_model_dir}"
    )


def main():

    args = parse_args()


    train_samples = parse_optional_int(
        args.train_samples
    )

    val_samples = parse_optional_int(
        args.val_samples
    )


    if args.epochs <= 0:

        raise ValueError(
            "epochs must be greater than 0."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "batch_size must be greater than 0."
        )

    if args.learning_rate <= 0:

        raise ValueError(
            "learning_rate must be greater than 0."
        )

    if args.weight_decay < 0:

        raise ValueError(
            "weight_decay cannot be negative."
        )

    if not 0 <= args.warmup_ratio < 1:

        raise ValueError(
            "warmup_ratio must be in [0, 1)."
        )

    if args.gradient_accumulation_steps <= 0:

        raise ValueError(
            "gradient_accumulation_steps must "
            "be greater than 0."
        )

    if args.max_grad_norm <= 0:

        raise ValueError(
            "max_grad_norm must be greater than 0."
        )

    if args.num_workers < 0:

        raise ValueError(
            "num_workers cannot be negative."
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(
        args.output_dir
    )

    paths = create_directories(
        output_dir
    )

    best_model_dir = paths[
        "best_model_dir"
    ]

    checkpoint_dir = paths[
        "checkpoint_dir"
    ]

    history_file = paths[
        "history_file"
    ]

    # Header
    print()
    print("=" * 75)
    print("LAYOUTLMV3 TRAINING")
    print("=" * 75)
    print(
        f"Model                    : "
        f"{MODEL_NAME}"
    )
    print(
        f"Device                   : "
        f"{device}"
    )
    print(
        f"Epochs                   : "
        f"{args.epochs}"
    )
    print(
        f"Batch size               : "
        f"{args.batch_size}"
    )
    print(
        f"Learning rate            : "
        f"{args.learning_rate}"
    )
    print(
        f"Weight decay             : "
        f"{args.weight_decay}"
    )
    print(
        f"Warmup ratio             : "
        f"{args.warmup_ratio}"
    )
    print(
        f"Gradient accumulation   : "
        f"{args.gradient_accumulation_steps}"
    )
    print(
        f"Max gradient norm        : "
        f"{args.max_grad_norm}"
    )
    print(
        f"Validation ratio         : "
        f"{args.val_ratio}"
    )
    print(
        f"Train samples            : "
        f"{train_samples if train_samples is not None else 'ALL'}"
    )
    print(
        f"Validation samples       : "
        f"{val_samples if val_samples is not None else 'ALL'}"
    )
    print(
        f"Random seed              : "
        f"{args.seed}"
    )
    print("=" * 75)


    #Dataset
    print()
    print(
        "Loading SROIE dataset..."
    )

    dataset = SROIEDataset(
        split="train",
        model_name=MODEL_NAME,
    )

    print(
        f"Full training dataset: "
        f"{len(dataset)} samples"
    )

    # Labels
    num_labels = len(
        dataset.label_names
    )

    print()
    print("Label configuration:")

    for idx, label in enumerate(
        dataset.label_names
    ):

        print(
            f"  {idx}: {label}"
        )

    print(
        f"Number of labels: "
        f"{num_labels}"
    )

    #Split
    print()
    print(
        "Creating train/validation split..."
    )

    train_dataset, val_dataset = (
        create_train_val_split(
            dataset=dataset,
            val_ratio=args.val_ratio,
            train_samples=train_samples,
            val_samples=val_samples,
            seed=args.seed,
        )
    )

    print(
        f"Training samples  : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # Save configuration
    save_training_config(
        args=args,
        dataset=dataset,
        output_dir=output_dir,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        device=device,
    )

    #DataLoaders

    print()
    print(
        "Creating DataLoaders..."
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"Training batches  : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    #Trainer

    print()
    print(
        "Creating LayoutLMTrainer..."
    )

    trainer = LayoutLMTrainer(
        model_name=MODEL_NAME,
        num_labels=num_labels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=str(device),
        id2label=dataset.id2label,
        label2id=dataset.label2id,
        max_grad_norm=args.max_grad_norm,
    )

    print(
        "[OK] Trainer created."
    )

    # Scheduler
    updates_per_epoch = (
        len(train_loader)
        + args.gradient_accumulation_steps
        - 1
    ) // args.gradient_accumulation_steps

    total_training_steps = (
        updates_per_epoch
        * args.epochs
    )

    trainer.create_scheduler(
        total_training_steps=(
            total_training_steps
        ),
        warmup_ratio=args.warmup_ratio,
    )

    print(
        f"[OK] Scheduler created."
    )

    print(
        f"Total optimizer steps: "
        f"{total_training_steps}"
    )

    # Training state
    history = []

    best_entity_f1 = -float(
        "inf"
    )

    epochs_without_improvement = 0

    start_epoch = 1

    # Resume
    if args.resume is not None:

        print()
        print(
            "[RESUME] Loading checkpoint..."
        )

        checkpoint = load_checkpoint(
            trainer=trainer,
            checkpoint_path=Path(
                args.resume
            ),
        )

        previous_epoch = (
            checkpoint["epoch"]
        )

        start_epoch = (
            previous_epoch + 1
        )

        best_entity_f1 = checkpoint.get(
            "best_entity_f1",
            -float("inf"),
        )

        epochs_without_improvement = (
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        history = checkpoint.get(
            "history",
            [],
        )

        print(
            f"Resuming after epoch: "
            f"{previous_epoch}"
        )

        print(
            f"Best Entity F1: "
            f"{best_entity_f1:.4f}"
        )

    #Training
    print()
    print(
        "Starting training..."
    )

    training_start = time.time()

    # Epoch loop
    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        print()
        print(
            f"---------- "
            f"Epoch {epoch}/{args.epochs} "
            f"----------"
        )

        epoch_start = time.time()

        # Train
        train_loss = trainer.train_epoch(
            dataloader=train_loader,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
        )

        # Validation
        metrics = trainer.evaluate(
            val_loader
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # Print
        print_epoch_metrics(
            epoch=epoch,
            total_epochs=args.epochs,
            train_loss=train_loss,
            metrics=metrics,
            epoch_time=epoch_time,
        )

        # History
        epoch_result = {
            "epoch": epoch,

            "train_loss": float(
                train_loss
            ),

            "validation": metrics,

            "epoch_time_seconds":
                float(epoch_time),
        }

        history.append(
            epoch_result
        )

        save_history(
            history=history,
            history_file=history_file,
        )

        # Best model
        current_entity_f1 = float(
            metrics["entity_f1"]
        )

        improved = (
            current_entity_f1
            >
            best_entity_f1
            + MIN_DELTA
        )

        if improved:

            best_entity_f1 = (
                current_entity_f1
            )

            epochs_without_improvement = 0

            print()
            print(
                "[BEST MODEL]"
            )

            print(
                f"Entity F1 improved to "
                f"{best_entity_f1:.4f}"
            )

            save_best_model(
                trainer=trainer,
                dataset=dataset,
                best_model_dir=best_model_dir,
                epoch=epoch,
                metrics=metrics,
                args=args,
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "No Entity F1 improvement."
            )

            print(
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # Checkpoint
        checkpoint_path = (
            checkpoint_dir
            / f"checkpoint_epoch_{epoch}.pt"
        )

        save_checkpoint(
            trainer=trainer,
            epoch=epoch,
            metrics=metrics,
            best_entity_f1=best_entity_f1,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            history=history,
            path=checkpoint_path,
            args=args,
        )

        print(
            f"[OK] Checkpoint saved: "
            f"{checkpoint_path}"
        )

        # Last checkpoint
        last_checkpoint_path = (
            checkpoint_dir
            / "last_checkpoint.pt"
        )

        save_checkpoint(
            trainer=trainer,
            epoch=epoch,
            metrics=metrics,
            best_entity_f1=best_entity_f1,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            history=history,
            path=last_checkpoint_path,
            args=args,
        )

        # Early stopping
        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "=" * 75
            )

            print(
                "EARLY STOPPING"
            )

            print(
                f"No Entity F1 improvement "
                f"for "
                f"{EARLY_STOPPING_PATIENCE} "
                f"epochs."
            )

            print(
                "=" * 75
            )

            break

    #Finished
    total_training_time = (
        time.time()
        - training_start
    )
    print()
    print(
        "Training finished."
    )
    print()
    print("=" * 75)
    print("TRAINING SUMMARY")
    print("=" * 75)
    print(
        f"Best Entity F1      : "
        f"{best_entity_f1:.4f}"
    )
    print(
        f"Best model          : "
        f"{best_model_dir}"
    )
    print(
        f"Checkpoints         : "
        f"{checkpoint_dir}"
    )
    print(
        f"History             : "
        f"{history_file}"
    )
    print(
        f"Config              : "
        f"{output_dir / 'training_config.json'}"
    )
    print(
        f"Total training time : "
        f"{total_training_time:.2f} sec"
    )
    print("=" * 75)
    print()
    print(
        "SUCCESS!"
    )

if __name__ == "__main__":
    main()