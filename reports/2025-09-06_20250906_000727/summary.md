# Bot Detection Report - 2025-09-06

**Generated on:** 2025-09-06 00:07:27

## Executive Summary

- **Total Cohorts Analyzed**: 39
- **High Risk Cohorts**: 0
- **Medium Risk Cohorts**: 20
- **Low Risk Cohorts**: 17

## Risk Distribution

| Risk Band | Count | Percentage |
|-----------|-------|------------|
| High      | 0 | 0.0% |
| Medium    | 20 | 51.3% |
| Low       | 17 | 43.6% |

## Score Statistics

### Botness Scores
- **Mean**: 0.549
- **Std**: 0.147
- **Min**: 0.177
- **Max**: 0.718

### Rule Scores
- **Mean**: 0.419
- **Std**: 0.031

### Anomaly Scores
- **Mean**: 0.680
- **Std**: 0.280

## Top High Risk Cohorts

No high risk cohorts found.

## Top Medium Risk Cohorts

| site | partner | ip | botness | risk_band | n_events | night_share | top_signature_share | ladder_sweep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| booking_br | booking | 190.115.65.140 | 0.718 | Medium | 5 | 100.0 | 0.0 | False |
| booking_fr | booking | 37.165.22.192 | 0.713 | Medium | 5 | 100.0 | 0.0 | False |
| booking_de | booking | 212.51.10.110 | 0.713 | Medium | 5 | 100.0 | 0.0 | False |
| booking_nl | booking | 77.167.160.110 | 0.712 | Medium | 5 | 100.0 | 0.0 | False |
| booking_se | booking | 90.238.43.18 | 0.699 | Medium | 5 | 100.0 | 0.0 | False |
| booking_us | booking | 157.191.45.86 | 0.698 | Medium | 5 | 100.0 | 0.0 | False |
| booking_de | booking | 2.241.249.182 | 0.689 | Medium | 5 | 100.0 | 0.0 | False |
| booking_de | booking | 178.202.190.183 | 0.687 | Medium | 5 | 100.0 | 0.0 | False |
| booking_de | booking | 87.78.237.177 | 0.681 | Medium | 6 | 100.0 | 0.0 | False |
| booking_mt | booking | 77.71.138.69 | 0.672 | Medium | 5 | 100.0 | 0.0 | False |

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
*Report generated on 2025-09-06 00:07:27*
