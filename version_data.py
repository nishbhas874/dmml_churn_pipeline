#!/usr/bin/env python3
"""
Wrapper script for data versioning.
Runs the actual versioning script from src/versioning/
"""

import os

if __name__ == "__main__":
    os.system("python src/versioning/version_data.py")
