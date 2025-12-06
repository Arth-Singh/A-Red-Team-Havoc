#!/usr/bin/env python3
"""
A-Red-Team-Havoc - Dashboard Runner
Run from project root: python run_dashboard.py
"""

import sys
import os

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from dashboard.app import run_dashboard

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A-Red-Team-Havoc Dashboard")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')

    args = parser.parse_args()
    run_dashboard(args.host, args.port, args.debug)
