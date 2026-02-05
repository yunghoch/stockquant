import pandas as pd
import numpy as np
from lasps.utils.helpers import compute_label, normalize_minmax
from lasps.utils.metrics import classification_metrics


def test_compute_label_buy():
    prices = pd.Series([100.0, 101, 102, 103, 104, 110])
    assert compute_label(prices, 0) == 2


def test_compute_label_sell():
    prices = pd.Series([100.0, 99, 98, 97, 96, 90])
    assert compute_label(prices, 0) == 0


def test_compute_label_hold():
    prices = pd.Series([100.0, 100, 100, 100, 100, 101])
    assert compute_label(prices, 0) == 1


def test_compute_label_insufficient_data():
    prices = pd.Series([100.0, 101, 102])
    assert compute_label(prices, 0) == -1


def test_normalize_minmax():
    s = pd.Series([0, 50, 100])
    result = normalize_minmax(s)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 1.0


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
