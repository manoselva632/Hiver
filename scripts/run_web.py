"""
Script: run_web.py
Launches the AI Support Agent interactive dashboard on a local port.
Usage:
    python scripts/run_web.py [--port 8000]
"""

import os
import sys
import argparse

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Launch AI Support Agent Web Application.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind web server (default: 8000)")
    args = parser.parse_args()

    run_server(port=args.port)


if __name__ == "__main__":
    main()
