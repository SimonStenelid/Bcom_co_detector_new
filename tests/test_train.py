"""Tests for supervised training and scoring."""

import numpy as np
import pandas as pd

from botminer.config import Config
from botminer.train import train_supervised
from botminer.scoring import _compute_supervised_scores
from botminer.heuristics import apply_heuristics


def test_train_and_score(tmp_path):
    config = Config.from_file("config.yaml")
    config.io.models_dir = str(tmp_path)

    data = pd.DataFrame({
        "n_events": [10, 120, 80, 15],
        "repeat_ratio": [0.1, 0.95, 0.93, 0.2],
        "top_signature_share": [0.2, 0.96, 0.95, 0.3],
        "cookie_rate": [0.9, 0.05, 0.08, 0.85],
        "referrer_rate": [0.8, 0.05, 0.05, 0.9],
    })

    labeled = apply_heuristics(data)
    train_supervised(labeled, config)

    scores = _compute_supervised_scores(data, config)
    assert scores is not None
    assert (tmp_path / "model.pkl").exists()
    bots = labeled["is_bot"].values
    assert scores[bots].mean() > scores[~bots].mean()
    assert np.all(scores >= 0) and np.all(scores <= 1)
