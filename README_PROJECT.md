# TrafficAI - Helmet Detection Project

A complete YOLOv8-based helmet detection system for traffic safety monitoring.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Project Structure
```bash
python setup_project.py
```

### 3. Prepare Your Dataset

#### Option A: Use XML Annotations
1. Place your images in `data/images/`
2. Place XML annotations in `data/annotations/`
3. Convert to YOLO format:
```bash
python app/xml_to_yolo_converter.py
```

#### Option B: Manual YOLO Labels
1. Place images in `data/images/`
2. Create corresponding `.txt` label files in `data/labels/`

### 4. Train the Model
```bash
python train_model.py
```

### 5. Run Detection
```bash
# Single image
python app/main.py --image data/samples/your_image.jpg

# Batch processing
python app/main.py --folder data/samples/
```

## 📁 Project Structure

```
TrafficAI/
├── app/                          # Main application code
│   ├── __init__.py
│   ├── main.py                   # Main entry point
│   ├── helmet_detector.py        # YOLOv8 detection logic
│   └── xml_to_yolo_converter.py  # Annotation converter
├── config/
│   └── dataset.yaml              # Dataset configuration
├── data/
│   ├── images/                   # Training images
│   ├── labels/                   # YOLO format labels
│   ├── annotations/              # XML annotations (optional)
│   ├── samples/                  # Test images
│   └── masks/                    # Lane masks (future feature)
├── models/
│   └── helmet_yolov8.pt          # Trained model (after training)
├── requirements.txt              # Python dependencies
├── train_model.py               # Training script
└── setup_project.py             # Project setup utility
```

## 🎯 Features

### ✅ Current Features
- **Helmet Detection**: Detects riders with/without helmets using YOLOv8
- **XML to YOLO Conversion**: Convert Pascal VOC annotations to YOLO format
- **Batch Processing**: Process multiple images at once
- **Visual Results**: Annotated images with bounding boxes
- **Training Pipeline**: Complete model training workflow

### 🔮 Future Features (from original design)
- **Lane Violation Detection**: Using binary masks to detect wrong lane usage
- **Real-time Video Processing**: Process video streams
- **Multi-camera Support**: Handle multiple camera feeds

## 🏷️ Class Labels

The model is trained to detect two classes:
- **Class 0**: `Without Helmet` (Red bounding box)
- **Class 1**: `With Helmet` (Green bounding box)

## 🔧 Configuration

### Model Settings
- **Model**: YOLOv8 Nano (lightweight and fast)
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.5 (adjustable)

### Training Parameters
- **Epochs**: 100
- **Batch Size**: 16
- **Patience**: 15 (early stopping)
- **Device**: Auto-detect (GPU if available, CPU otherwise)

## 📊 Usage Examples

### Basic Detection
```bash
python app/main.py --image motorcycle.jpg
```

### Custom Model and Confidence
```bash
python app/main.py --image test.jpg --model models/custom_model.pt --confidence 0.7
```

### Training with Custom Dataset
```bash
python train_model.py
```

## 🛠️ Development

### Adding New Features
1. Helmet detection logic is in `app/helmet_detector.py`
2. Main application flow is in `app/main.py`
3. Training configuration is in `config/dataset.yaml`

### Extending the System
- Add new detection classes by updating the `class_mapping` in the converter
- Modify training parameters in `train_model.py`
- Add post-processing logic in `helmet_detector.py`

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM
- GPU recommended but not required

### Python Dependencies
- ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- torch >= 2.0.0
- numpy, PIL, PyYAML

## 🤝 Integration with Original TrafficAI

This project integrates seamlessly with the existing TrafficAI framework:
- Uses the same YOLOv8 approach mentioned in `AGENTS.md`
- Follows the structure outlined in the original `README.md`
- Can be extended with lane detection using the mask approach

## 🐛 Troubleshooting

### Common Issues

**"No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**"CUDA out of memory"**
- Reduce batch size in training
- Use CPU instead: change device to 'cpu' in training script

**"No images found"**
- Check that images are in the correct directory
- Ensure image formats are supported (jpg, png, bmp)

**Poor detection results**
- Increase training epochs
- Add more training data
- Adjust confidence threshold

## 📜 License

This project is part of TrafficAI and follows the same licensing terms.
