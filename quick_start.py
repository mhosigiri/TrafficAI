#!/usr/bin/env python3
"""
Quick Start Script for TrafficAI
Run this to start monitoring with default settings.
"""

import subprocess
import sys

print("Starting TrafficAI Traffic Monitor...")
print("Press Ctrl+C to stop\n")

# Start with camera monitoring
subprocess.run([sys.executable, "deploy_traffic_monitor.py", "--mode", "camera"])
