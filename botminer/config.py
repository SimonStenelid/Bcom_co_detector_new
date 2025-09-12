"""Configuration management for botminer."""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class IngestConfig:
    """Configuration for data ingestion."""
    tz: str = "UTC"
    min_events_per_cohort: int = 30


@dataclass
class FeaturesConfig:
    """Configuration for feature engineering."""
    night_hours: List[int]
    gap_percentiles: List[int]
    ladder_max_gap_days: int = 3
    min_gap_seconds: float = 0.1
    max_gap_seconds: float = 3600


@dataclass
class ScoringConfig:
    """Configuration for scoring algorithms."""
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    alpha_blend_iso: float = 0.5
    beta_supervised: float = 0.5


@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest."""
    n_estimators: int = 300
    contamination: float = 0.05
    max_samples: float = 0.8
    random_state: int = 42


@dataclass
class HistoryConfig:
    """Configuration for historical data processing."""
    lookback_days: int = 30
    ewma_alpha_7d: float = 0.3
    ewma_alpha_30d: float = 0.1


@dataclass
class IOConfig:
    """Configuration for input/output paths."""
    input_dir: str = "data/raw"
    features_dir: str = "data/features"
    history_dir: str = "data/history"
    reports_dir: str = "reports"
    models_dir: str = "data/models"


@dataclass
class ReportingConfig:
    """Configuration for report generation."""
    top_n_cohorts: int = 50
    sample_rows_per_cohort: int = 3
    include_evidence: bool = True


@dataclass
class Config:
    """Main configuration class."""
    ingest: IngestConfig
    features: FeaturesConfig
    scoring: ScoringConfig
    isolation_forest: IsolationForestConfig
    history: HistoryConfig
    io: IOConfig
    reporting: ReportingConfig

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            ingest=IngestConfig(**data.get('ingest', {})),
            features=FeaturesConfig(**data.get('features', {})),
            scoring=ScoringConfig(**data.get('scoring', {})),
            isolation_forest=IsolationForestConfig(**data.get('isolation_forest', {})),
            history=HistoryConfig(**data.get('history', {})),
            io=IOConfig(**data.get('io', {})),
            reporting=ReportingConfig(**data.get('reporting', {}))
        )

    def get_paths(self, date_str: str) -> Dict[str, Path]:
        """Get file paths for a given date."""
        base_path = Path(".")
        return {
            'raw': base_path / self.io.input_dir / f"{date_str}.parquet",
            'features': base_path / self.io.features_dir / f"{date_str}.parquet",
            'history': base_path / self.io.history_dir / f"{date_str}.parquet",
            'reports_dir': base_path / self.io.reports_dir / date_str,
            'cohorts_csv': base_path / self.io.reports_dir / date_str / "cohorts.csv",
            'cohorts_json': base_path / self.io.reports_dir / date_str / "cohorts.json",
            'summary_md': base_path / self.io.reports_dir / date_str / "summary.md",
        }
