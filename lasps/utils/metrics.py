import numpy as np
from typing import Dict


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = (y_true == y_pred).mean()
    per_class = {}
    for cls in range(3):
        mask_true = y_true == cls
        mask_pred = y_pred == cls
        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}
