"""Dependency-free classification metrics for security PoCs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classification_metrics(
    gold: list[str],
    predicted: list[str],
    labels: tuple[str, ...],
) -> ClassificationMetrics:
    if not gold or len(gold) != len(predicted):
        raise ValueError("gold and predicted must be non-empty and equally sized")
    allowed = set(labels)
    if any(value not in allowed for value in gold + predicted):
        raise ValueError("gold or predicted contains an unknown label")

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        false_positive = sum(
            g != label and p == label for g, p in zip(gold, predicted, strict=True)
        )
        false_negative = sum(
            g == label and p != label for g, p in zip(gold, predicted, strict=True)
        )
        support = sum(value == label for value in gold)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return ClassificationMetrics(
        accuracy=sum(g == p for g, p in zip(gold, predicted, strict=True)) / len(gold),
        macro_precision=sum(precisions) / len(precisions),
        macro_recall=sum(recalls) / len(recalls),
        macro_f1=sum(f1_scores) / len(f1_scores),
        per_class=per_class,
    )
