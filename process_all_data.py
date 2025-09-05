#!/usr/bin/env python3
"""
Process all data files in the data/raw folder with batch mode.
Usage: python process_all_data.py <date>
"""

import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_all_data.py <date>")
        print("Example: python process_all_data.py 2025-09-05")
        sys.exit(1)
    
    date = sys.argv[1]
    
    # Check if data directory exists and has files
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    booking_files = list(data_dir.glob("booking_*.csv"))
    ip_files = list(data_dir.glob("On_demand_report_*.csv"))
    
    if not booking_files:
        print("Error: No booking files found in data/raw directory")
        sys.exit(1)
    
    if not ip_files:
        print("Error: No IP files found in data/raw directory")
        sys.exit(1)
    
    print(f"Processing all data for {date}...")
    print(f"Found {len(booking_files)} booking files:")
    for f in booking_files:
        print(f"  - {f.name}")
    print(f"Found {len(ip_files)} IP files:")
    for f in ip_files:
        print(f"  - {f.name}")
    print()
    
    # Run the pipeline in batch mode
    cmd = [
        "python", "-m", "botminer.cli", "pipeline",
        date,
        "--batch",
        "--verbose"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ Successfully processed all data for {date}")
        print(f"📊 Reports available in: reports/{date}/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error processing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
