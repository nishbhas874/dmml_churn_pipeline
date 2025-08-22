#!/usr/bin/env python3
"""
Wrapper script for feature store management.
Runs the actual feature store script from src/feature_store/
"""

import os

if __name__ == "__main__":
    os.system("python src/feature_store/manage_features.py")
