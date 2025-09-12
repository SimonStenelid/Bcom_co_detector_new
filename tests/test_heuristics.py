"""Tests for heuristic bot labeling."""

import pandas as pd

from botminer.heuristics import apply_heuristics


def test_apply_heuristics_flags_bot():
    data = pd.DataFrame({
        "n_events": [10, 120, 60],
        "repeat_ratio": [0.1, 0.95, 0.95],
        "top_signature_share": [0.2, 0.96, 0.5],
        "cookie_rate": [0.9, 0.05, 0.05],
        "referrer_rate": [0.8, 0.05, 0.05],
    })
    labeled = apply_heuristics(data)
    assert not labeled.loc[0, "is_bot"]
    assert bool(labeled.loc[1, "is_bot"])
    assert bool(labeled.loc[2, "is_bot"])
