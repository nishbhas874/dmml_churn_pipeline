#!/usr/bin/env python3
"""
Wrapper script for data ingestion.
Runs the actual ingestion script from src/ingestion/
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import and run the actual ingestion script
if __name__ == "__main__":
    os.system("python src/ingestion/get_data.py")
