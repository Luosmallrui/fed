from __future__ import annotations

from collections.abc import Iterable

import torch


def primary_metric_name(task: str) -> str:
    if task == "mnli":
        return "mean_matched_mismatched_accuracy"
    if task in {"mrpc", "qqp"}:
        return "glue_accuracy_f1_mean"
    return "accuracy"


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    predictions = predictions.long().flatten()
    labels = labels.long().flatten()
    valid = (labels >= 0) & (labels < num_labels)
    encoded = labels[valid] * num_labels + predictions[valid]
    return torch.bincount(encoded, minlength=num_labels * num_labels).reshape(num_labels, num_labels)


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.double()
    total = matrix.sum()
    accuracy = matrix.diag().sum() / total.clamp_min(1)
    tp = matrix.diag()
    precision = tp / matrix.sum(dim=0).clamp_min(1)
    recall = tp / matrix.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    result = {
        "accuracy": float(accuracy.item()),
        "macro_f1": float(f1.mean().item()),
    }
    if matrix.shape == (2, 2):
        positive_f1 = float(f1[1].item())
        result["f1"] = positive_f1
        result["glue_score"] = 0.5 * (result["accuracy"] + positive_f1)
    return result


def primary_metric(task: str, split_metrics: dict[str, dict[str, float]]) -> float:
    if "train_holdout" in split_metrics:
        values = split_metrics["train_holdout"]
        if task in {"mrpc", "qqp"}:
            return 0.5 * (values["accuracy"] + values["f1"])
        return values["accuracy"]
    if task == "mnli":
        return 0.5 * (
            split_metrics["validation_matched"]["accuracy"]
            + split_metrics["validation_mismatched"]["accuracy"]
        )
    values = split_metrics["validation"]
    if task in {"mrpc", "qqp"}:
        return 0.5 * (values["accuracy"] + values["f1"])
    return values["accuracy"]


def aggregate_client_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(metrics)
    if not values:
        return {}
    return {
        "mean_client_accuracy": sum(x["accuracy"] for x in values) / len(values),
        "worst_client_accuracy": min(x["accuracy"] for x in values),
        "mean_client_macro_f1": sum(x["macro_f1"] for x in values) / len(values),
        "worst_client_macro_f1": min(x["macro_f1"] for x in values),
    }
