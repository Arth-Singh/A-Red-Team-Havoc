#!/usr/bin/env python3
"""
A-Red-Team-Havoc - Quick Runner Script
Run from project root: python run_havoc.py
"""

import sys
import os

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from src.main import main

if __name__ == "__main__":
    main()
