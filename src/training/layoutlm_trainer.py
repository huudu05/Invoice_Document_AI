from pathlib import Path
from typing import Dict, Optional, Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    LayoutLMv3ForTokenClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


class LayoutLMTrainer:
    """
    Trainer for LayoutLMv3 token classification.

    Responsibilities
    ----------------
    - Load LayoutLMv3
    - Train one epoch
    - Validation
    - Prediction
    - Token-level evaluation
    - Entity-token evaluation
    - Gradient clipping
    - Learning-rate scheduling
    - Model saving/loading
    """

    def __init__(
        self,
        model: Optional[LayoutLMv3ForTokenClassification] = None,
        num_labels: int = 5,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        model_name: str = "microsoft/layoutlmv3-base",
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        max_grad_norm: float = 1.0,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm

        # Label mapping
        if id2label is None:

            id2label = {
                0: "S-COMPANY",
                1: "S-DATE",
                2: "S-ADDRESS",
                3: "S-TOTAL",
                4: "O",
            }

        if label2id is None:

            label2id = {
                label: idx
                for idx, label in id2label.items()
            }

        self.id2label = {
            int(key): value
            for key, value in id2label.items()
        }

        self.label2id = label2id

        if len(self.id2label) != num_labels:
            raise ValueError(
                "Number of labels in id2label does not "
                "match num_labels."
            )

        if len(self.label2id) != num_labels:
            raise ValueError(
                "Number of labels in label2id does not "
                "match num_labels."
            )

        # Model
        if model is None:

            self.model = (
                LayoutLMv3ForTokenClassification
                .from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label=self.id2label,
                    label2id=self.label2id,
                )
            )

        else:
            self.model = model

        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = None


    # Scheduler
    def create_scheduler(
        self,
        total_training_steps: int,
        warmup_ratio: float = 0.1,
    ) -> None:

        if total_training_steps <= 0:
            raise ValueError(
                "total_training_steps must be greater than 0."
            )

        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(
                "warmup_ratio must be in [0, 1)."
            )

        warmup_steps = int(
            total_training_steps * warmup_ratio
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

    # Move batch
    def _move_batch_to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        return {
            key: value.to(self.device)
            for key, value in batch.items()
        }

    # Forward
    def _forward(
        self,
        batch: Dict[str, torch.Tensor],
        include_labels: bool = True,
    ):
        """
        Run LayoutLMv3 forward pass.
        """

        if include_labels:

            return self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )

        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            pixel_values=batch["pixel_values"],
        )

    # Train one epoch
    def train_epoch(
        self,
        dataloader: DataLoader,
        gradient_accumulation_steps: int = 1,
    ) -> float:
        """
        Train the model for one complete epoch.

        Returns
        -------
        float
            Average training loss.
        """

        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be "
                "greater than 0."
            )

        self.model.train()

        total_loss = 0.0
        num_steps = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(
            dataloader,
            start=1,
        ):

            batch = self._move_batch_to_device(
                batch
            )

            outputs = self._forward(
                batch,
                include_labels=True,
            )

            loss = outputs.loss

            total_loss += loss.item()
            num_steps += 1

            loss_for_backward = (
                loss / gradient_accumulation_steps
            )

            loss_for_backward.backward()

            should_update = (
                step % gradient_accumulation_steps == 0
                or step == len(dataloader)
            )

            if should_update:

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm,
                )

                # Optimizer
                self.optimizer.step()

                # Scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            # Progress
            print(
                f"Step {step:04d}/{len(dataloader):04d} | "
                f"Loss: {loss.item():.6f}",
                end="\r",
            )

        print()

        if num_steps == 0:
            raise RuntimeError(
                "Training DataLoader is empty."
            )

        return total_loss / num_steps

    # Backward compatibility
    def train_one_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:

        return self.train_epoch(
            dataloader
        )

    # Validation
    def validation_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:
        self.model.eval()

        total_loss = 0.0
        num_steps = 0

        with torch.no_grad():

            for batch in dataloader:

                batch = self._move_batch_to_device(
                    batch
                )

                outputs = self._forward(
                    batch,
                    include_labels=True,
                )

                total_loss += outputs.loss.item()
                num_steps += 1

        if num_steps == 0:
            raise RuntimeError(
                "Validation DataLoader is empty."
            )

        return total_loss / num_steps

    # Prediction
    def predict(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        
        self.model.eval()

        batch = self._move_batch_to_device(
            batch
        )

        with torch.no_grad():

            outputs = self._forward(
                batch,
                include_labels=False,
            )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )

        return predictions

    # Evaluation
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, Any]:

        self.model.eval()

        all_predictions = []
        all_labels = []

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():

            for batch in dataloader:

                batch = self._move_batch_to_device(
                    batch
                )

                outputs = self._forward(
                    batch,
                    include_labels=True,
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

                for pred, label in zip(
                    predictions,
                    labels,
                ):

                    for p, l in zip(
                        pred,
                        label,
                    ):

                        label_value = l.item()

                        # Ignore special tokens and padding.
                        if label_value == -100:
                            continue

                        all_predictions.append(
                            p.item()
                        )

                        all_labels.append(
                            label_value
                        )

        if num_batches == 0:
            raise RuntimeError(
                "Evaluation DataLoader is empty."
            )

        if len(all_labels) == 0:
            raise RuntimeError(
                "No valid labels found during evaluation."
            )

        # Accuracy
        accuracy = accuracy_score(
            all_labels,
            all_predictions,
        )

        # Per-label metrics
        label_ids = list(
            range(self.num_labels)
        )

        (
            precision,
            recall,
            f1,
            support,
        ) = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=label_ids,
            zero_division=0,
        )

        # Macro
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        # Weighted
        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _,
        ) = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )

        # Entity-token metrics
        entity_label_ids = [
            label_id
            for label_id in label_ids
            if self.id2label[label_id] != "O"
        ]

        (
            entity_precision,
            entity_recall,
            entity_f1,
            entity_support,
        ) = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=entity_label_ids,
            average="macro",
            zero_division=0,
        )

        # O metrics
        o_label_id = self.label2id.get("O")

        if o_label_id is not None:

            o_index = label_ids.index(
                o_label_id
            )

            o_precision = precision[o_index]
            o_recall = recall[o_index]
            o_f1 = f1[o_index]

        else:

            o_precision = 0.0
            o_recall = 0.0
            o_f1 = 0.0

        # Per-label
        per_label = {}

        for label_id in label_ids:

            label_name = self.id2label[
                label_id
            ]

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

        # Learning rate
        current_learning_rate = (
            self.optimizer.param_groups[0]["lr"]
        )

        # Result
        metrics = {
            "loss": float(
                total_loss / num_batches
            ),

            "accuracy": float(
                accuracy
            ),

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

            "o_precision": float(
                o_precision
            ),

            "o_recall": float(
                o_recall
            ),

            "o_f1": float(
                o_f1
            ),

            "num_tokens": int(
                len(all_labels)
            ),

            "per_label": per_label,

            "learning_rate": float(
                current_learning_rate
            ),
        }

        return metrics

    # Print metrics
    def print_metrics(
        self,
        metrics: Dict[str, Any],
    ) -> None:

        print("\n" + "-" * 70)
        print("EVALUATION METRICS")
        print("-" * 70)

        print(
            f"Loss             : "
            f"{metrics['loss']:.6f}"
        )

        print(
            f"Accuracy         : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Macro Precision  : "
            f"{metrics['macro_precision']:.4f}"
        )

        print(
            f"Macro Recall     : "
            f"{metrics['macro_recall']:.4f}"
        )

        print(
            f"Macro F1         : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1      : "
            f"{metrics['weighted_f1']:.4f}"
        )

        print()

        print(
            f"Entity Precision : "
            f"{metrics['entity_precision']:.4f}"
        )

        print(
            f"Entity Recall    : "
            f"{metrics['entity_recall']:.4f}"
        )

        print(
            f"Entity F1        : "
            f"{metrics['entity_f1']:.4f}"
        )

        print()

        print(
            f"O Precision      : "
            f"{metrics['o_precision']:.4f}"
        )

        print(
            f"O Recall         : "
            f"{metrics['o_recall']:.4f}"
        )

        print(
            f"O F1             : "
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

        print("\nPer-label metrics:")

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

        print("-" * 70)

    # Save model
    def save_model(
        self,
        output_dir: str,
    ) -> None:

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(
            output_path
        )

        print(
            f"[OK] Model saved to: "
            f"{output_path}"
        )

    # Load model
    def load_model(
        self,
        model_path: str,
    ) -> None:

        self.model = (
            LayoutLMv3ForTokenClassification
            .from_pretrained(
                model_path
            )
        )

        self.model.to(
            self.device
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = None

        print(
            f"[OK] Model loaded from: "
            f"{model_path}"
        )