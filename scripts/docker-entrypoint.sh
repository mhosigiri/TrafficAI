#!/bin/bash
set -e

# Function to check GPU availability
check_gpu() {
    python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
}

# Function to check Tesseract
check_tesseract() {
    tesseract --version > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Tesseract OCR: Available"
    else
        echo "Tesseract OCR: Not Available"
    fi
}

echo "TrafficAI Container Starting..."
echo "================================"

# Check system status
echo "System Check:"
check_gpu
check_tesseract

# Check if models exist
if [ -f "/app/models/helmet_yolov8_gpu.pt" ]; then
    echo "Helmet Model: Found"
else
    echo "Helmet Model: Not Found"
fi

if [ -f "/app/models/license_plate_yolov8.pt" ]; then
    echo "License Plate Model: Found"
else
    echo "License Plate Model: Not Found (will use base model)"
fi

echo "================================"

# Execute the command passed to docker run
exec "$@"
