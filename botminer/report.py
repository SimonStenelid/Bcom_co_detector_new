"""Report generation module for bot detection results."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import Config
from .scoring import load_scored_features, get_top_risk_cohorts, compute_score_statistics


def generate_reports(date: str, config: Config) -> Dict[str, Any]:
    """
    Generate comprehensive reports for bot detection results.
    
    Args:
        date: Date string in YYYY-MM-DD format
        config: Configuration object
        
    Returns:
        Dictionary with report generation results
    """
    logger = logging.getLogger(__name__)
    
    # Load scored features
    features_df = load_scored_features(date, config)
    
    # Create timestamped reports directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = config.get_paths(date)['reports_dir'].parent / f"{date}_{timestamp}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CSV report
    csv_path = _generate_csv_report(features_df, date, config, reports_dir)
    
    # Generate JSON report
    json_path = _generate_json_report(features_df, date, config, reports_dir)
    
    # Generate Markdown summary
    md_path = _generate_markdown_report(features_df, date, config, reports_dir)
    
    logger.info(f"Generated reports for {len(features_df)} cohorts")
    
    return {
        'csv_path': str(csv_path),
        'json_path': str(json_path),
        'md_path': str(md_path),
        'total_cohorts': len(features_df)
    }


def _generate_csv_report(features_df: pd.DataFrame, date: str, config: Config, reports_dir: Path) -> Path:
    """Generate CSV report with all cohort data."""
    logger = logging.getLogger(__name__)
    
    # Select columns for CSV report
    csv_columns = [
        'cohort_key', 'date', 'site', 'partner', 'ip', 'user_agent',
        'ua_full', 'ua_full_mode_share', 'ua_full_ip_diversity', 'ua_full_ip_diversity_ratio',
        'rule_score', 'iso_score', 'botness', 'risk_band',
        'n_events', 'night_share', 'top_signature_share', 'repeat_ratio',
        'gap_p50', 'max_run_lt_1s', 'cookie_rate', 'referrer_rate',
        'http11_share', 'http2_share', 'ladder_sweep', 'hourly_entropy'
    ]
    
    # Filter to available columns
    available_columns = [col for col in csv_columns if col in features_df.columns]
    csv_df = features_df[available_columns].copy()
    
    # Sort by botness score
    csv_df = csv_df.sort_values('botness', ascending=False)
    
    # Save CSV
    csv_path = reports_dir / 'cohorts.csv'
    csv_df.to_csv(csv_path, index=False)
    
    logger.info(f"Generated CSV report: {csv_path}")
    
    return csv_path


def _generate_json_report(features_df: pd.DataFrame, date: str, config: Config, reports_dir: Path) -> Path:
    """Generate JSON report with detailed cohort information."""
    logger = logging.getLogger(__name__)
    
    # Get top risk cohorts
    high_risk = get_top_risk_cohorts(features_df, 'High', config.reporting.top_n_cohorts)
    medium_risk = get_top_risk_cohorts(features_df, 'Medium', config.reporting.top_n_cohorts)
    
    # Compute statistics
    stats = compute_score_statistics(features_df)
    
    # Prepare JSON data
    json_data = {
        'metadata': {
            'date': date,
            'generated_at': datetime.now().isoformat(),
            'total_cohorts': len(features_df),
            'config': {
                'scoring_weights': config.scoring.weights,
                'risk_thresholds': config.scoring.thresholds,
                'alpha_blend': config.scoring.alpha_blend_iso
            }
        },
        'statistics': stats,
        'high_risk_cohorts': _format_cohorts_for_json(high_risk),
        'medium_risk_cohorts': _format_cohorts_for_json(medium_risk),
        'summary': {
            'high_risk_count': len(high_risk),
            'medium_risk_count': len(medium_risk),
            'low_risk_count': len(features_df[features_df['risk_band'] == 'Low'])
        }
    }
    
    # Save JSON
    json_path = reports_dir / 'cohorts.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    logger.info(f"Generated JSON report: {json_path}")
    
    return json_path


def _generate_markdown_report(features_df: pd.DataFrame, date: str, config: Config, reports_dir: Path) -> Path:
    """Generate Markdown summary report."""
    logger = logging.getLogger(__name__)
    
    # Get top cohorts
    high_risk = get_top_risk_cohorts(features_df, 'High', 20)
    medium_risk = get_top_risk_cohorts(features_df, 'Medium', 20)
    
    # Compute statistics
    stats = compute_score_statistics(features_df)
    
    # Generate markdown content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = _create_markdown_content(
        features_df, high_risk, medium_risk, stats, date, config, timestamp
    )
    
    # Save markdown
    md_path = reports_dir / 'summary.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    logger.info(f"Generated Markdown report: {md_path}")
    
    return md_path


def _format_cohorts_for_json(cohorts_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format cohort data for JSON output."""
    if len(cohorts_df) == 0:
        return []
    
    # Select key columns
    key_columns = [
        'cohort_key', 'site', 'partner', 'ip', 'user_agent',
        'ua_full', 'ua_full_mode_share', 'ua_full_ip_diversity', 'ua_full_ip_diversity_ratio',
        'rule_score', 'iso_score', 'botness', 'risk_band',
        'n_events', 'night_share', 'top_signature_share', 'repeat_ratio',
        'gap_p50', 'max_run_lt_1s', 'cookie_rate', 'referrer_rate',
        'ladder_sweep', 'hourly_entropy'
    ]
    
    available_columns = [col for col in key_columns if col in cohorts_df.columns]
    
    # Convert to list of dictionaries
    cohorts_list = []
    for _, row in cohorts_df[available_columns].iterrows():
        cohort_dict = row.to_dict()
        # Convert numpy types to Python types
        for key, value in cohort_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                cohort_dict[key] = value.item()
            elif isinstance(value, np.bool_):
                cohort_dict[key] = bool(value)
        cohorts_list.append(cohort_dict)
    
    return cohorts_list


def _create_markdown_content(
    features_df: pd.DataFrame,
    high_risk: pd.DataFrame,
    medium_risk: pd.DataFrame,
    stats: Dict[str, Any],
    date: str,
    config: Config,
    timestamp: str
) -> str:
    """Create markdown content for the summary report."""
    
    # Handle empty dataframe case
    if len(features_df) == 0:
        return f"""# Bot Detection Report - {date}

**Generated on:** {timestamp}

## Executive Summary

- **Total Cohorts Analyzed**: 0
- **High Risk Cohorts**: 0
- **Medium Risk Cohorts**: 0
- **Low Risk Cohorts**: 0

## No Data Available

No cohorts were processed for this date. This could be due to:
- No data available for the specified date
- All cohorts had fewer than the minimum required events
- Data processing errors

---
*Report generated on {timestamp}*
"""
    
    content = f"""# Bot Detection Report - {date}

**Generated on:** {timestamp}

## Executive Summary

- **Total Cohorts Analyzed**: {len(features_df):,}
- **High Risk Cohorts**: {len(high_risk):,}
- **Medium Risk Cohorts**: {len(medium_risk):,}
- **Low Risk Cohorts**: {len(features_df[features_df['risk_band'] == 'Low']):,}

## Risk Distribution

| Risk Band | Count | Percentage |
|-----------|-------|------------|
| High      | {len(high_risk):,} | {len(high_risk)/len(features_df)*100:.1f}% |
| Medium    | {len(medium_risk):,} | {len(medium_risk)/len(features_df)*100:.1f}% |
| Low       | {len(features_df[features_df['risk_band'] == 'Low']):,} | {len(features_df[features_df['risk_band'] == 'Low'])/len(features_df)*100:.1f}% |

## Score Statistics

### Botness Scores
- **Mean**: {stats['score_statistics']['botness']['mean']:.3f}
- **Std**: {stats['score_statistics']['botness']['std']:.3f}
- **Min**: {stats['score_statistics']['botness']['min']:.3f}
- **Max**: {stats['score_statistics']['botness']['max']:.3f}

### Rule Scores
- **Mean**: {stats['score_statistics']['rule_score']['mean']:.3f}
- **Std**: {stats['score_statistics']['rule_score']['std']:.3f}

### Anomaly Scores
- **Mean**: {stats['score_statistics']['iso_score']['mean']:.3f}
- **Std**: {stats['score_statistics']['iso_score']['std']:.3f}

## Top High Risk Cohorts

"""
    
    if len(high_risk) > 0:
        content += _create_cohort_table(high_risk.head(10))
    else:
        content += "No high risk cohorts found.\n"
    
    content += "\n## Top Medium Risk Cohorts\n\n"
    
    if len(medium_risk) > 0:
        content += _create_cohort_table(medium_risk.head(10))
    else:
        content += "No medium risk cohorts found.\n"
    
    # Add IP diversity by full user agent section
    try:
        ua_div = features_df[[
            'ua_full', 'ua_full_ip_diversity', 'ua_full_ip_diversity_ratio'
        ]].dropna().drop_duplicates()
        if len(ua_div) > 0:
            ua_div = ua_div.sort_values('ua_full_ip_diversity', ascending=False).head(10)
            content += "\n## IP Diversity by Full User Agent\n\n"
            content += "Top UA strings by number of unique IPs observed for this date. High values can indicate automated clients shared across many IPs.\n\n"
            # Build table
            header_cols = ['ua_full', 'unique_ips', 'ip_share']
            content += "| ua_full | unique_ips | ip_share |\n"
            content += "| --- | --- | --- |\n"
            for _, row in ua_div.iterrows():
                ua = str(row.get('ua_full', 'Unknown'))
                unique_ips = int(row.get('ua_full_ip_diversity', 0))
                share = float(row.get('ua_full_ip_diversity_ratio', 0.0))
                content += f"| {ua} | {unique_ips} | {share:.3f} |\n"
    except Exception:
        # Don't break report generation for this section
        pass
    
    content += f"""
## Configuration

- **Scoring Weights**: {config.scoring.weights}
- **Risk Thresholds**: {config.scoring.thresholds}
- **Alpha Blend**: {config.scoring.alpha_blend_iso}

## Methodology

This report uses a combination of rule-based scoring and anomaly detection to identify potentially automated traffic:

1. **Rule-based Scoring**: Evaluates patterns in timing, repetition, client behavior, and volume
2. **Anomaly Detection**: Uses Isolation Forest to identify unusual patterns
3. **Combined Score**: Weighted combination of rule-based and anomaly scores
4. **Risk Bands**: Cohorts are classified as High, Medium, or Low risk based on thresholds

## Key Indicators

- **High Night Activity**: Automated traffic often occurs during off-peak hours
- **Signature Repetition**: Bots tend to repeat similar search patterns
- **Low Client Hygiene**: Missing cookies, referrers, or unusual HTTP patterns
- **Rapid Requests**: Very short time gaps between requests
- **Ladder Sweeps**: Systematic date progression patterns

---
*Report generated on {timestamp}*
"""

    return content


def _create_cohort_table(cohorts_df: pd.DataFrame) -> str:
    """Create a markdown table for cohort data."""
    if len(cohorts_df) == 0:
        return "No cohorts to display.\n"
    
    # Select key columns for display
    display_columns = [
        'site', 'partner', 'ip', 'botness', 'risk_band',
        'n_events', 'night_share', 'top_signature_share', 'ladder_sweep'
    ]
    
    available_columns = [col for col in display_columns if col in cohorts_df.columns]
    display_df = cohorts_df[available_columns].copy()
    
    # Format values for display
    if 'botness' in display_df.columns:
        display_df['botness'] = display_df['botness'].round(3)
    if 'night_share' in display_df.columns:
        display_df['night_share'] = (display_df['night_share'] * 100).round(1)
    if 'top_signature_share' in display_df.columns:
        display_df['top_signature_share'] = (display_df['top_signature_share'] * 100).round(1)
    
    # Create markdown table
    table_lines = []
    
    # Header
    header = "| " + " | ".join(display_df.columns) + " |"
    table_lines.append(header)
    
    # Separator
    separator = "|" + "|".join([" --- " for _ in display_df.columns]) + "|"
    table_lines.append(separator)
    
    # Rows
    for _, row in display_df.iterrows():
        row_str = "| " + " | ".join([str(val) for val in row.values]) + " |"
        table_lines.append(row_str)
    
    return "\n".join(table_lines) + "\n"


def display_summary_console(features_df: pd.DataFrame, date: str) -> None:
    """Display a summary in the console using Rich."""
    console = Console()
    
    # Create summary table
    table = Table(title=f"Bot Detection Summary - {date}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Add summary rows
    table.add_row("Total Cohorts", f"{len(features_df):,}")
    table.add_row("High Risk", f"{len(features_df[features_df['risk_band'] == 'High']):,}")
    table.add_row("Medium Risk", f"{len(features_df[features_df['risk_band'] == 'Medium']):,}")
    table.add_row("Low Risk", f"{len(features_df[features_df['risk_band'] == 'Low']):,}")
    
    if 'botness' in features_df.columns:
        table.add_row("Avg Botness Score", f"{features_df['botness'].mean():.3f}")
        table.add_row("Max Botness Score", f"{features_df['botness'].max():.3f}")
    
    console.print(table)
    
    # Show top high risk cohorts
    high_risk = get_top_risk_cohorts(features_df, 'High', 5)
    if len(high_risk) > 0:
        console.print("\n[bold red]Top High Risk Cohorts:[/bold red]")
        for _, row in high_risk.iterrows():
            console.print(f"  • {row['site']}/{row['partner']} - {row['ip']} (Score: {row['botness']:.3f})")


def export_evidence_samples(
    features_df: pd.DataFrame,
    date: str,
    config: Config,
    raw_data_path: Optional[str] = None
) -> Dict[str, Any]:
    """Export sample evidence for top risk cohorts."""
    logger = logging.getLogger(__name__)
    
    if raw_data_path is None:
        raw_data_path = config.get_paths(date)['raw']
    
    if not Path(raw_data_path).exists():
        logger.warning(f"Raw data not found for evidence samples: {raw_data_path}")
        return {'samples_exported': 0}
    
    # Load raw data
    raw_df = pd.read_parquet(raw_data_path)
    
    # Get top risk cohorts
    high_risk = get_top_risk_cohorts(features_df, 'High', 10)
    
    evidence_samples = []
    
    for _, cohort in high_risk.iterrows():
        cohort_key = cohort['cohort_key']
        cohort_data = raw_df[raw_df['cohort_key'] == cohort_key]
        
        if len(cohort_data) > 0:
            # Sample a few rows as evidence
            sample_size = min(config.reporting.sample_rows_per_cohort, len(cohort_data))
            sample_data = cohort_data.sample(n=sample_size, random_state=42)
            
            evidence_samples.append({
                'cohort_key': cohort_key,
                'botness_score': cohort['botness'],
                'risk_band': cohort['risk_band'],
                'sample_rows': sample_data.to_dict('records')
            })
    
    # Save evidence samples
    evidence_path = config.get_paths(date)['reports_dir'] / "evidence_samples.json"
    with open(evidence_path, 'w') as f:
        json.dump(evidence_samples, f, indent=2, default=str)
    
    logger.info(f"Exported evidence samples for {len(evidence_samples)} cohorts")
    
    return {
        'samples_exported': len(evidence_samples),
        'evidence_path': str(evidence_path)
    }
