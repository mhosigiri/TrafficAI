#!/bin/bash

echo "========================================"
echo "    TrafficAI Web Application"
echo "========================================"
echo ""
echo "Starting TrafficAI Web Application..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python setup_trafficai.py"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if web_app.py exists
if [ ! -f "web_app.py" ]; then
    echo "❌ web_app.py not found!"
    echo "Please ensure you're in the correct directory"
    exit 1
fi

# Start the application
python web_app.py
