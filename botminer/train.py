"""Training utilities for supervised bot classifiers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
import pickle

from .config import Config
from .heuristics import apply_heuristics

logger = logging.getLogger(__name__)


def train_supervised(
    features_df: pd.DataFrame,
    config: Config,
    label_col: str = "is_bot",
    model_path: Optional[Path] = None,
) -> Path:
    """Train a Gradient Boosting classifier and persist it."""
    if label_col not in features_df.columns:
        features_df = apply_heuristics(features_df)
    labeled = features_df.dropna(subset=[label_col])
    if labeled.empty:
        raise ValueError("No labeled rows available for training")

    y = labeled[label_col].astype(int)
    numeric_cols = labeled.select_dtypes(include=[np.number]).columns
    exclude = [label_col, 'rule_score', 'iso_score', 'botness', 'risk_band']
    feature_cols = [c for c in numeric_cols if c not in exclude]
    X = labeled[feature_cols].fillna(0)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_scaled, y)

    if model_path is None:
        model_dir = Path(config.io.models_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
    else:
        model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }, f)

    logger.info(f"Saved supervised model to {model_path}")
    return model_path


def train_from_files(dates: List[str], config: Config, label_col: str = "is_bot") -> Path:
    frames = []
    for d in dates:
        path = config.get_paths(d)["features"]
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("No feature files found for training")
    all_features = pd.concat(frames, ignore_index=True)
    return train_supervised(all_features, config, label_col=label_col)
