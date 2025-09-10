# Bot Detection Report - 2025-09-09

**Generated on:** 2025-09-10 13:43:24

## Executive Summary

- **Total Cohorts Analyzed**: 45
- **High Risk Cohorts**: 0
- **Medium Risk Cohorts**: 20
- **Low Risk Cohorts**: 14

## Risk Distribution

| Risk Band | Count | Percentage |
|-----------|-------|------------|
| High      | 0 | 0.0% |
| Medium    | 20 | 44.4% |
| Low       | 14 | 31.1% |

## Score Statistics

### Botness Scores
- **Mean**: 0.585
- **Std**: 0.140
- **Min**: 0.120
- **Max**: 0.747

### Rule Scores
- **Mean**: 0.441
- **Std**: 0.076

### Anomaly Scores
- **Mean**: 0.730
- **Std**: 0.233

## Top High Risk Cohorts

No high risk cohorts found.

## Top Medium Risk Cohorts

| site | partner | ip | botness | risk_band | n_events | night_share | top_signature_share | ladder_sweep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| booking_co | booking | 45.169.98.83 | 0.747 | Medium | 240 | 100.0 | 0.0 | False |
| booking_co | booking | 8.243.163.246 | 0.743 | Medium | 263 | 100.0 | 0.0 | False |
| booking_co | booking | 201.184.74.106 | 0.743 | Medium | 377 | 100.0 | 0.0 | False |
| booking_co | booking | 186.148.180.112 | 0.738 | Medium | 346 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.36.109 | 0.736 | Medium | 42 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.33.67 | 0.733 | Medium | 54 | 100.0 | 0.0 | False |
| booking_co | booking | 190.85.174.10 | 0.728 | Medium | 242 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.34.30 | 0.725 | Medium | 20 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.38.37 | 0.722 | Medium | 92 | 100.0 | 0.0 | False |
| booking_co | booking | 185.215.54.36 | 0.714 | Medium | 14 | 100.0 | 0.0 | False |

## Configuration

- **Scoring Weights**: {'cadence': 0.2, 'repetition': 0.3, 'speed': 0.2, 'hygiene': 0.15, 'protocol': 0.1, 'volume': 0.05}
- **Risk Thresholds**: {'high': 0.75, 'medium': 0.55, 'low': 0.25}
- **Alpha Blend**: 0.5

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
*Report generated on 2025-09-10 13:43:24*
