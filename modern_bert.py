"""
Fine-tune ModernBERT for RAGTruth span-level hallucination detection.

Input:
    context + response

Training target:
    token labels on response tokens only:
        O / B-HALL / I-HALL

The context tokens are kept as evidence, but ignored in the loss with -100.

Example:
    python train_modernbert_span_detector.py --max-train-samples 500 --epochs 1

Full run:
    python train_modernbert_span_detector.py --epochs 3 --batch-size 1 --grad-accum 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL_LIST = ["O", "B-HALL", "I-HALL"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
IGNORE_INDEX = -100





def parse_hallucination_labels(raw: Any) -> list[dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def merge_spans(labels: list[dict[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for label in labels:
        try:
            start = int(label["start"])
            end = int(label["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            spans.append((start, end))

    if not spans:
        return []

    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def token_label_for_response_offset(
    token_start: int,
    token_end: int,
    hallucination_spans: list[tuple[int, int]],
    active_span_idx: int | None,
) -> tuple[int, int | None]:
    if token_end <= token_start:
        return LABEL2ID["O"], active_span_idx

    for span_idx, (span_start, span_end) in enumerate(hallucination_spans):
        overlaps = token_start < span_end and token_end > span_start
        if not overlaps:
            continue
        if active_span_idx != span_idx:
            return LABEL2ID["B-HALL"], span_idx
        return LABEL2ID["I-HALL"], span_idx

    return LABEL2ID["O"], None


class RagTruthSpanDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        response_col: str,
        label_col: str,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_col = response_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        context = str(row["context"])
        response = str(row[self.response_col])
        hallucination_spans = merge_spans(parse_hallucination_labels(row.get(self.label_col)))

        encoded = self.tokenizer(
            context,
            response,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        labels: list[int] = []
        active_span_idx: int | None = None
        sequence_ids = encoded.sequence_ids()

        for seq_id, (token_start, token_end) in zip(sequence_ids, encoded["offset_mapping"]):
            if seq_id != 1:
                labels.append(IGNORE_INDEX)
                active_span_idx = None
                continue

            label_id, active_span_idx = token_label_for_response_offset(
                int(token_start),
                int(token_end),
                hallucination_spans,
                active_span_idx,
            )
            labels.append(label_id)

        encoded.pop("offset_mapping")
        encoded["labels"] = labels
        return encoded


class WeightedTokenClassificationTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_ragtruth_frame(path: Path, response_col: str, label_col: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    missing = {"context", response_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def sample_frame(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def compute_class_weights(
    df: pd.DataFrame,
    label_col: str,
    o_weight: float,
    hall_weight: float,
) -> torch.Tensor:
    has_hall = df[label_col].apply(lambda raw: bool(merge_spans(parse_hallucination_labels(raw))))
    hall_rows = int(has_hall.sum())
    clean_rows = int((~has_hall).sum())
    print(f"Training rows with hallucination labels: {hall_rows}")
    print(f"Training rows without hallucination labels: {clean_rows}")
    return torch.tensor([o_weight, hall_weight, hall_weight], dtype=torch.float)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != IGNORE_INDEX

    gold_hall = np.isin(labels, [LABEL2ID["B-HALL"], LABEL2ID["I-HALL"]]) & mask
    pred_hall = np.isin(preds, [LABEL2ID["B-HALL"], LABEL2ID["I-HALL"]]) & mask

    tp = int(np.logical_and(gold_hall, pred_hall).sum())
    fp = int(np.logical_and(~gold_hall, pred_hall & mask).sum())
    fn = int(np.logical_and(gold_hall, ~pred_hall).sum())
    tn = int(np.logical_and(~gold_hall, ~pred_hall & mask).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0

    return {
        "hall_token_precision": precision,
        "hall_token_recall": recall,
        "hall_token_f1": f1,
        "token_accuracy": accuracy,
    }


def spans_from_token_predictions(
    response: str,
    offsets: list[tuple[int, int]],
    pred_ids: list[int],
    sequence_ids: list[int | None],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current_start: int | None = None
    current_end: int | None = None

    for seq_id, (start, end), pred_id in zip(sequence_ids, offsets, pred_ids):
        if seq_id != 1:
            continue

        label = ID2LABEL[int(pred_id)]
        if label == "B-HALL":
            if current_start is not None and current_end is not None:
                spans.append(
                    {
                        "start": current_start,
                        "end": current_end,
                        "text": response[current_start:current_end],
                    }
                )
            current_start, current_end = int(start), int(end)
        elif label == "I-HALL" and current_start is not None:
            current_end = int(end)
        else:
            if current_start is not None and current_end is not None:
                spans.append(
                    {
                        "start": current_start,
                        "end": current_end,
                        "text": response[current_start:current_end],
                    }
                )
            current_start, current_end = None, None

    if current_start is not None and current_end is not None:
        spans.append(
            {
                "start": current_start,
                "end": current_end,
                "text": response[current_start:current_end],
            }
        )

    return spans


@torch.no_grad()
def predict_hallucination_spans(
    model: AutoModelForTokenClassification,
    tokenizer: Any,
    context: str,
    response: str,
    max_length: int,
    device: str,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        context,
        response,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    logits = model(**encoded).logits[0]
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
    return spans_from_token_predictions(response, offsets, pred_ids, sequence_ids)


def save_demo_predictions(
    model: AutoModelForTokenClassification,
    tokenizer: Any,
    df: pd.DataFrame,
    args: argparse.Namespace,
    response_col: str,
    label_col: str,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    demo_rows = []
    for _, row in df.head(args.demo_predictions).iterrows():
        context = str(row["context"])
        response = str(row[response_col])
        pred_spans = predict_hallucination_spans(
            model=model,
            tokenizer=tokenizer,
            context=context,
            response=response,
            max_length=args.max_length,
            device=device,
        )
        gold_spans = merge_spans(parse_hallucination_labels(row.get(label_col)))
        demo_rows.append(
            {
                "id": row.get("id", ""),
                "gold_spans": json.dumps(
                    [
                        {"start": start, "end": end, "text": response[start:end]}
                        for start, end in gold_spans
                    ],
                    ensure_ascii=False,
                ),
                "predicted_spans": json.dumps(pred_spans, ensure_ascii=False),
            }
        )

    output_path = Path(args.output_dir) / "demo_predictions.csv"
    pd.DataFrame(demo_rows).to_csv(output_path, index=False)
    print(f"Wrote demo predictions to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ModernBERT for RAGTruth span detection.")
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--train-file", default="train-00000-of-00001.parquet")
    parser.add_argument("--eval-file", default="test-00000-of-00001.parquet")
    parser.add_argument("--output-dir", default="modernbert_ragtruth_span_detector")
    parser.add_argument("--response-col", default="output")
    parser.add_argument("--label-col", default="hallucination_labels")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--o-weight", type=float, default=1.0)
    parser.add_argument("--hall-weight", type=float, default=8.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--demo-predictions", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> None:
    
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print(f"Loading train data: {train_path}")
    train_df = load_ragtruth_frame(train_path, args.response_col, args.label_col)
    print(f"Loading eval data: {eval_path}")
    eval_df = load_ragtruth_frame(eval_path, args.response_col, args.label_col)

    if "quality" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df["quality"].astype(str).str.lower() == "good"].reset_index(drop=True)
        print(f"Kept good-quality train rows: {len(train_df)} / {before}")
    if "quality" in eval_df.columns:
        before = len(eval_df)
        eval_df = eval_df[eval_df["quality"].astype(str).str.lower() == "good"].reset_index(drop=True)
        print(f"Kept good-quality eval rows: {len(eval_df)} / {before}")

    train_df = sample_frame(train_df, args.max_train_samples, args.seed)
    eval_df = sample_frame(eval_df, args.max_eval_samples, args.seed)
    print(f"Training rows: {len(train_df)}")
    print(f"Evaluation rows: {len(eval_df)}")

    train_dataset = RagTruthSpanDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        response_col=args.response_col,
        label_col=args.label_col,
    )
    eval_dataset = RagTruthSpanDataset(
        df=eval_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        response_col=args.response_col,
        label_col=args.label_col,
    )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class_weights = compute_class_weights(
        train_df,
        label_col=args.label_col,
        o_weight=args.o_weight,
        hall_weight=args.hall_weight,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="hall_token_f1",
        greater_is_better=True,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
    )

    trainer = WeightedTokenClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("Starting training...")
    trainer.train()

    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model and tokenizer to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    if args.demo_predictions > 0:
        save_demo_predictions(
            model=model,
            tokenizer=tokenizer,
            df=eval_df,
            args=args,
            response_col=args.response_col,
            label_col=args.label_col,
        )


if __name__ == "__main__":
    main()
