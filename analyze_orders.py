#!/usr/bin/env python3
"""Analyze order patterns in the processed data."""

import pandas as pd
import sys
from pathlib import Path

def analyze_orders(date: str = "2025-09-05"):
    """Analyze order patterns in the processed data."""
    
    # Load the processed data
    data_path = f"data/raw/{date}.parquet"
    if not Path(data_path).exists():
        print(f"Error: Processed data not found: {data_path}")
        return
    
    print(f"Analyzing order patterns for date: {date}")
    print("=" * 50)
    
    # Load data
    df = pd.read_parquet(data_path)
    
    print(f"Total records: {len(df):,}")
    print()
    
    # Order statistics
    if 'num_orders' in df.columns:
        total_orders = df['num_orders'].sum()
        events_with_orders = (df['num_orders'] > 0).sum()
        conversion_rate = events_with_orders / len(df) * 100
        
        print("Order Statistics:")
        print("-" * 30)
        print(f"Total orders:           {total_orders:6,}")
        print(f"Events with orders:     {events_with_orders:6,} ({conversion_rate:5.1f}%)")
        print(f"Events without orders:  {len(df) - events_with_orders:6,} ({100-conversion_rate:5.1f}%)")
        print(f"Average orders/event:   {total_orders/len(df):6.3f}")
        print()
        
        # Order distribution
        print("Order Distribution:")
        print("-" * 30)
        order_counts = df['num_orders'].value_counts().sort_index()
        for orders, count in order_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"{int(orders):2d} orders: {count:6,} events ({percentage:5.1f}%)")
        print()
        
        # Order patterns by user agent type
        if 'user_agent_type' in df.columns:
            print("Order Patterns by User Agent Type:")
            print("-" * 40)
            ua_order_stats = df.groupby('user_agent_type').agg({
                'num_orders': ['count', 'sum', 'mean'],
                'has_order': 'mean'
            }).round(3)
            
            for ua_type in ua_order_stats.index:
                total_events = ua_order_stats.loc[ua_type, ('num_orders', 'count')]
                total_orders_ua = ua_order_stats.loc[ua_type, ('num_orders', 'sum')]
                avg_orders = ua_order_stats.loc[ua_type, ('num_orders', 'mean')]
                conversion_rate_ua = ua_order_stats.loc[ua_type, ('has_order', 'mean')] * 100
                
                print(f"{ua_type:15} {total_events:6,} events, {total_orders_ua:6,} orders, {avg_orders:6.3f} avg, {conversion_rate_ua:5.1f}% conv")
            print()
        
        # Order patterns by browser
        if 'user_agent' in df.columns:
            print("Order Patterns by Browser:")
            print("-" * 30)
            browser_order_stats = df.groupby('user_agent').agg({
                'num_orders': ['count', 'sum', 'mean'],
                'has_order': 'mean'
            }).round(3)
            
            for browser in browser_order_stats.index:
                total_events = browser_order_stats.loc[browser, ('num_orders', 'count')]
                total_orders_browser = browser_order_stats.loc[browser, ('num_orders', 'sum')]
                avg_orders = browser_order_stats.loc[browser, ('num_orders', 'mean')]
                conversion_rate_browser = browser_order_stats.loc[browser, ('has_order', 'mean')] * 100
                
                print(f"{browser:15} {total_events:6,} events, {total_orders_browser:6,} orders, {avg_orders:6.3f} avg, {conversion_rate_browser:5.1f}% conv")
            print()
        
        # Suspicious order patterns (potential bot indicators)
        print("Suspicious Order Patterns (Potential Bot Indicators):")
        print("-" * 55)
        
        # Cohorts with all same order values
        cohort_order_stats = df.groupby('cohort_key')['num_orders'].agg(['count', 'nunique', 'min', 'max', 'sum'])
        suspicious_cohorts = cohort_order_stats[
            (cohort_order_stats['nunique'] == 1) & 
            (cohort_order_stats['count'] >= 5)  # At least 5 events
        ]
        
        print(f"Cohorts with identical order patterns: {len(suspicious_cohorts)}")
        if len(suspicious_cohorts) > 0:
            print("Top suspicious cohorts:")
            for cohort, stats in suspicious_cohorts.head(5).iterrows():
                print(f"  {cohort}: {stats['count']} events, all with {stats['min']} orders")
        print()
        
        # Cohorts with only 0 or 1 orders
        zero_one_cohorts = cohort_order_stats[
            (cohort_order_stats['min'] == 0) & 
            (cohort_order_stats['max'] <= 1) & 
            (cohort_order_stats['count'] >= 5)
        ]
        print(f"Cohorts with only 0-1 orders: {len(zero_one_cohorts)}")
        print()
        
        # High-volume, low-conversion cohorts
        high_volume_low_conv = cohort_order_stats[
            (cohort_order_stats['count'] >= 20) &  # High volume
            (cohort_order_stats['sum'] == 0)  # No orders
        ]
        print(f"High-volume, zero-conversion cohorts: {len(high_volume_low_conv)}")
        if len(high_volume_low_conv) > 0:
            print("Top high-volume, zero-conversion cohorts:")
            for cohort, stats in high_volume_low_conv.head(5).iterrows():
                print(f"  {cohort}: {stats['count']} events, 0 orders")
        
    else:
        print("No order data found in the dataset")

def main():
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = "2025-09-05"
    
    analyze_orders(date)

if __name__ == "__main__":
    main()
