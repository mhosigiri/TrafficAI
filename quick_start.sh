#!/bin/bash
# TrafficAI Helmet Detection - Quick Start Script

echo "🚀 TrafficAI Helmet Detection - Quick Start"
echo "==========================================="

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Set up project structure
echo "📁 Setting up project structure..."
python3 setup_project.py

# Test the conversion
echo "🔄 Testing XML to YOLO conversion..."
python3 app/xml_to_yolo_converter.py

# Check converted labels
echo "📊 Checking converted labels..."
if [ -f "data/labels/sample1.txt" ]; then
    echo "✅ Sample label created:"
    cat data/labels/sample1.txt
else
    echo "❌ Label conversion failed"
fi

echo ""
echo "✅ Quick start complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your images to data/images/"
echo "2. Add XML annotations to data/annotations/"
echo "3. Convert: python3 app/xml_to_yolo_converter.py"
echo "4. Train: python3 train_model.py"
echo "5. Detect: python3 app/main.py --image your_image.jpg"
echo ""
echo "📚 Read INSTALL.md for detailed instructions"
