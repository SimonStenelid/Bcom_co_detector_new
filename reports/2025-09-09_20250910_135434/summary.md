# Bot Detection Report - 2025-09-09

**Generated on:** 2025-09-10 13:54:34

## Executive Summary

- **Total Cohorts Analyzed**: 45
- **High Risk Cohorts**: 7
- **Medium Risk Cohorts**: 20
- **Low Risk Cohorts**: 9

## Risk Distribution

| Risk Band | Count | Percentage |
|-----------|-------|------------|
| High      | 7 | 15.6% |
| Medium    | 20 | 44.4% |
| Low       | 9 | 20.0% |

## Score Statistics

### Botness Scores
- **Mean**: 0.620
- **Std**: 0.139
- **Min**: 0.184
- **Max**: 0.792

### Rule Scores
- **Mean**: 0.518
- **Std**: 0.066

### Anomaly Scores
- **Mean**: 0.721
- **Std**: 0.232

## Top High Risk Cohorts

| site | partner | ip | botness | risk_band | n_events | night_share | top_signature_share | ladder_sweep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| booking_co | booking | 8.243.163.246 | 0.792 | High | 263 | 100.0 | 0.0 | False |
| booking_co | booking | 201.184.74.106 | 0.78 | High | 377 | 100.0 | 0.0 | False |
| booking_co | booking | 45.169.98.83 | 0.78 | High | 240 | 100.0 | 0.0 | False |
| booking_co | booking | 190.85.174.10 | 0.778 | High | 242 | 100.0 | 0.0 | False |
| booking_co | booking | 186.148.180.112 | 0.771 | High | 346 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.34.30 | 0.77 | High | 20 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.36.109 | 0.768 | High | 42 | 100.0 | 0.0 | False |

## Top Medium Risk Cohorts

| site | partner | ip | botness | risk_band | n_events | night_share | top_signature_share | ladder_sweep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| booking_co | booking | 191.156.33.67 | 0.74 | Medium | 54 | 100.0 | 0.0 | False |
| booking_co | booking | 181.50.102.4 | 0.74 | Medium | 532 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.38.37 | 0.739 | Medium | 92 | 100.0 | 0.0 | False |
| booking_co | booking | 190.27.129.126 | 0.738 | Medium | 21 | 100.0 | 0.0 | False |
| booking_co | booking | 191.156.43.125 | 0.73 | Medium | 17 | 100.0 | 0.0 | False |
| booking_co | booking | 185.215.54.36 | 0.73 | Medium | 14 | 100.0 | 0.0 | False |
| booking_co | booking | 181.78.22.138 | 0.725 | Medium | 286 | 100.0 | 0.0 | False |
| booking_co | booking | 38.52.156.232 | 0.717 | Medium | 385 | 100.0 | 0.0 | False |
| booking_co | booking | 181.131.20.171 | 0.705 | Medium | 596 | 100.0 | 0.0 | False |
| booking_co | booking | 45.171.183.121 | 0.683 | Medium | 258 | 0.0 | 0.0 | False |

## IP Diversity by Full User Agent

Top UA strings by number of unique IPs observed for this date. High values can indicate automated clients shared across many IPs.

| ua_full | unique_ips | ip_share |
| --- | --- | --- |
| Chrome.86/PC/Linux/Web Browser | 20 | 0.263 |
| Chrome.86/PC/Windows/Web Browser | 20 | 0.263 |
| Chrome.85/PC/Mac OS/Web Browser | 18 | 0.237 |
| Chrome.85/PC/Windows/Web Browser | 18 | 0.237 |
| Chrome.86/PC/Mac OS/Web Browser | 17 | 0.224 |
| Chrome.85/PC/Linux/Web Browser | 16 | 0.211 |
| Chrome.84/PC/Mac OS/Web Browser | 14 | 0.184 |
| Chrome.139/PC/Windows/Web Browser | 14 | 0.184 |
| Chrome.140/PC/Windows/Web Browser | 13 | 0.171 |
| Chrome.84/PC/Windows/Web Browser | 13 | 0.171 |

## Configuration

- **Scoring Weights**: {'cadence': 0.2, 'repetition': 0.3, 'speed': 0.2, 'hygiene': 0.15, 'protocol': 0.1, 'volume': 0.05, 'ua_full_concentration': 0.1, 'ua_full_ip_diversity': 0.1}
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
*Report generated on 2025-09-10 13:54:34*
