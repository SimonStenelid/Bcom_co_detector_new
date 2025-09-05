#!/usr/bin/env python3
"""
Simple script to process booking data with IP addresses.
Usage: python process_data.py <booking_csv> <ip_csv> <date>
"""

import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print("Usage: python process_data.py <booking_csv> <ip_csv> <date>")
        print("Example: python process_data.py data/raw/booking_.csv data/raw/ip_data.csv 2025-09-05")
        sys.exit(1)
    
    booking_csv = sys.argv[1]
    ip_csv = sys.argv[2]
    date = sys.argv[3]
    
    # Validate files exist
    if not Path(booking_csv).exists():
        print(f"Error: Booking CSV file not found: {booking_csv}")
        sys.exit(1)
    
    if not Path(ip_csv).exists():
        print(f"Error: IP CSV file not found: {ip_csv}")
        sys.exit(1)
    
    # Run the pipeline
    cmd = [
        "python", "-m", "botminer.cli", "pipeline",
        booking_csv, date,
        "--ip-csv", ip_csv,
        "--verbose"
    ]
    
    print(f"Processing data for {date}...")
    print(f"Booking CSV: {booking_csv}")
    print(f"IP CSV: {ip_csv}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ Successfully processed data for {date}")
        print(f"📊 Reports available in: reports/{date}/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error processing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
