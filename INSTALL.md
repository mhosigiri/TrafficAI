# TrafficAI - Installation Guide

## 🚀 Quick Installation & Setup

### Prerequisites
- Python 3.8+ (recommended: Python 3.10+)
- pip (Python package manager)
- 4GB+ RAM
- GPU optional but recommended for training

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Required Libraries:**
- `ultralytics>=8.0.0` - YOLOv8 framework
- `opencv-python>=4.8.0` - Computer vision operations
- `torch>=2.0.0` - PyTorch for deep learning
- `numpy>=1.24.0` - Numerical computations
- `Pillow>=9.5.0` - Image processing
- `PyYAML>=6.0` - Configuration files
- `matplotlib>=3.7.0` - Plotting and visualization

### Step 2: Set Up Project Structure

```bash
# Run the setup script to create directories and sample files
python3 setup_project.py
```

This creates:
- `data/images/` - Training images
- `data/labels/` - YOLO format labels
- `data/annotations/` - XML annotations (optional)
- `data/samples/` - Test images
- `models/` - Trained model weights
- `config/` - Configuration files

### Step 3: Verify Installation

```bash
# Test if all libraries are properly installed
python3 -c "import ultralytics, cv2, numpy as np, torch; print('✅ All dependencies installed successfully!')"
```

## 📊 Dataset Preparation

### Option A: Using XML Annotations (Pascal VOC format)

1. **Add your images:**
   ```bash
   # Copy your images to the data/images/ folder
   cp your_images/*.jpg data/images/
   ```

2. **Add XML annotations:**
   ```bash
   # Copy XML annotation files to data/annotations/
   cp your_annotations/*.xml data/annotations/
   ```

3. **Convert to YOLO format:**
   ```bash
   python3 app/xml_to_yolo_converter.py
   ```

### Option B: Direct YOLO Format

1. **Add images:** Place images in `data/images/`
2. **Add labels:** Create corresponding `.txt` files in `data/labels/`

**YOLO Label Format:**
```
class_id x_center y_center width height
```
Where all coordinates are normalized (0-1):
- `class_id`: 0 = Without Helmet, 1 = With Helmet
- `x_center, y_center`: Center of bounding box
- `width, height`: Dimensions of bounding box

**Example label file (`data/labels/image1.txt`):**
```
1 0.234375 0.312500 0.156250 0.208333
0 0.546875 0.416667 0.156250 0.208333
```

## 🏋️ Model Training

### Quick Training
```bash
python3 train_model.py
```

### Custom Training Parameters
Edit the training script or use YOLOv8 directly:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained model
results = model.train(
    data='config/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='auto'  # 'cpu' or 'cuda'
)
```

**Training Output:**
- Model weights: `runs/detect/helmet_detection/weights/best.pt`
- Training logs: `runs/detect/helmet_detection/`
- Automatic copy to: `models/helmet_yolov8.pt`

## 🔍 Running Detection

### Single Image
```bash
python3 app/main.py --image data/samples/your_image.jpg
```

### Batch Processing
```bash
python3 app/main.py --folder data/samples/
```

### Custom Parameters
```bash
python3 app/main.py --image test.jpg --confidence 0.7 --model models/custom_model.pt
```

## 🧪 Testing the Complete Pipeline

### 1. Test with Sample Data
```bash
# The setup script creates a sample XML annotation
python3 app/xml_to_yolo_converter.py

# Check the converted label
cat data/labels/sample1.txt
# Output: 1 0.234375 0.312500 0.156250 0.208333
#         0 0.546875 0.416667 0.156250 0.208333
```

### 2. Test Detection (without custom model)
```bash
# This will use YOLOv8 nano as fallback for person detection
python3 app/main.py --image data/samples/any_image.jpg
```

### 3. Full Pipeline Test
```bash
# 1. Add real helmet dataset images to data/images/
# 2. Add XML annotations to data/annotations/
# 3. Convert annotations
python3 app/xml_to_yolo_converter.py

# 4. Train model
python3 train_model.py

# 5. Test detection
python3 app/main.py --image data/samples/test_image.jpg
```

## 🛠️ Integration with Your Code

The project integrates your original Kaggle notebook code:

### XML to YOLO Converter (`app/xml_to_yolo_converter.py`)
- ✅ Implements your `convert_xml2yolo()` function
- ✅ Uses your `get_text()` and `convert_coordinates()` helpers
- ✅ Same class mapping: `{"Without Helmet": 0, "With Helmet": 1}`

### YAML Configuration (`config/dataset.yaml`)
- ✅ Same structure as your original `dataset.yaml`
- ✅ Class names: ["Without Helmet", "With Helmet"]
- ✅ 2 classes (nc: 2)

### Model Training (`train_model.py`)
- ✅ Uses YOLOv8 like your notebook
- ✅ Same training approach with ultralytics
- ✅ Automatic GPU/CPU detection

## 🔧 Troubleshooting

### Common Installation Issues

**1. "command not found: python"**
```bash
# Use python3 instead
python3 --version
```

**2. "No module named 'ultralytics'"**
```bash
pip3 install ultralytics
```

**3. "CUDA out of memory"**
```bash
# Edit train_model.py and change batch size:
batch=8  # Reduce from 16 to 8
# Or force CPU training:
device='cpu'
```

**4. "No images found"**
```bash
# Check directory structure
ls -la data/images/
# Ensure images have correct extensions (.jpg, .png, .jpeg)
```

### Performance Optimization

**For faster training:**
- Use GPU: `device='cuda'`
- Increase batch size: `batch=32`
- Use larger model: `yolov8s.pt` or `yolov8m.pt`

**For smaller models:**
- Use nano model: `yolov8n.pt`
- Reduce image size: `imgsz=416`
- Fewer epochs: `epochs=50`

## 📈 Expected Results

After training with a good helmet dataset:
- **With Helmet**: Green bounding boxes
- **Without Helmet**: Red bounding boxes
- **Confidence scores**: Displayed on each detection
- **Violation summary**: Printed in console
- **Annotated images**: Saved with "_annotated.jpg" suffix

## 🎯 Next Steps

1. **Collect More Data**: Add more helmet/no-helmet images for better accuracy
2. **Fine-tune Model**: Adjust training parameters based on results
3. **Add Video Support**: Extend to process video files
4. **Deploy Model**: Use for real-time traffic monitoring
5. **Lane Detection**: Implement the lane mask feature from original design

## 📞 Support

If you encounter issues:
1. Check this installation guide
2. Verify all dependencies are installed
3. Ensure correct file paths and formats
4. Test with sample data first before using custom dataset
