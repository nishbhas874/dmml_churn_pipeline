#!/usr/bin/env python3
"""
Wrapper script for raw data storage.
Runs the actual storage script from src/storage/
"""

import os

if __name__ == "__main__":
    os.system("python src/storage/store_data.py")
