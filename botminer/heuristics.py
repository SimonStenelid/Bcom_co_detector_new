"""Rule-based heuristics for preliminary bot labeling."""

from __future__ import annotations

import pandas as pd


def apply_heuristics(features_df: pd.DataFrame) -> pd.DataFrame:
    """Label cohorts with simple rule-based heuristics.

    Parameters
    ----------
    features_df:
        DataFrame containing cohort level features. Expected columns
        include ``n_events``, ``repeat_ratio``, ``top_signature_share``,
        ``cookie_rate``, and ``referrer_rate``. Missing columns are treated
        as zeros which effectively disable the corresponding rule.

    Returns
    -------
    DataFrame
        Original frame with an additional boolean column ``is_bot`` that
        indicates whether any heuristic flagged the cohort as bot-like.
    """
    df = features_df.copy()

    for col in [
        'n_events', 'repeat_ratio', 'top_signature_share',
        'cookie_rate', 'referrer_rate'
    ]:
        if col not in df.columns:
            df[col] = 0

    rule_repeated_route = (
        (df['n_events'] >= 60) &
        (df['repeat_ratio'] > 0.9) &
        (df['top_signature_share'] > 0.9)
    )

    rule_low_hygiene = (
        (df['n_events'] >= 50) &
        (df['cookie_rate'] < 0.1) &
        (df['referrer_rate'] < 0.1)
    )

    df['is_bot'] = rule_repeated_route | rule_low_hygiene

    return df
