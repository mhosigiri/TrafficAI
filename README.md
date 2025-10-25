# TrafficAI - Intelligent Traffic Monitoring System

A real-time traffic monitoring system that uses computer vision and deep learning to detect helmet violations, wrong-lane violations, and license plate recognition.

## Features

- **Helmet Detection**: Automatically identifies riders without helmets using YOLOv8
- **Wrong Lane Detection**: Flags vehicles in prohibited zones using camera-specific binary masks
- **License Plate Recognition**: Detects and extracts license plate text using YOLO + Tesseract OCR
- **Multi-Mode Processing**: Support for images, videos, batch processing, and live camera feeds
- **Web Interface**: Modern, responsive web UI for easy interaction
- **Docker Support**: Containerized deployment with GPU support

## Demo

<p align="center">
  <img src="static/demo.gif" alt="TrafficAI Demo" width="800"/>
</p>

## Quick Start

### Prerequisites

- Python 3.10 or newer
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster inference

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TrafficAI.git
cd TrafficAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install tesseract-ocr libtesseract-dev
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
```bash
choco install tesseract
```

4. Download pre-trained models and place them in the `models/` directory:
   - `helmet_yolov8.pt` - Helmet detection model
   - `lp_yolov8.pt` - License plate detection model

### Basic Usage

**Process a single image:**
```bash
python app/main.py --image data/samples/sample1.jpg
```

**Process images in batch:**
```bash
python app/main.py --folder data/samples
```

**Enable license plate detection:**
```bash
python app/main.py --image data/samples/sample1.jpg --plates
```

## Web Application

Launch the web interface for an interactive experience:

```bash
python web_app.py
```
# This is pranil making subtle change to the readme file
Then open your browser to `http://localhost:3000`

### Web Features
- Drag-and-drop file upload
- Real-time processing visualization
- Video timeline analysis
- Detection statistics and reports
- Mobile-responsive design

## Docker Deployment

### CPU Version
```bash
docker build -t trafficai:cpu .
docker run -it -p 3000:3000 trafficai:cpu
```

### GPU Version
```bash
docker build -f Dockerfile.gpu -t trafficai:gpu .
docker run --gpus all -it -p 3000:3000 trafficai:gpu
```

### Docker Compose
```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## Project Structure

```
TrafficAI/
├── app/                      # Core application logic
│   ├── main.py              # Main entry point
│   ├── inference.py         # YOLOv8 inference wrapper
│   ├── plates.py            # License plate detection & OCR
│   ├── lane.py              # Wrong lane detection
│   └── draw_masks.py        # Mask creation utility
├── models/                   # Pre-trained models (user-provided)
│   ├── helmet_yolov8.pt
│   └── lp_yolov8.pt
├── data/                     # Data and masks
│   ├── samples/             # Test images
│   └── masks/               # Lane masks
├── static/                   # Web UI assets
├── templates/                # HTML templates
├── web_app.py               # Flask web application
├── train_model.py           # Model training script
└── requirements.txt         # Python dependencies
```

## How It Works

### Helmet Detection
Uses YOLOv8 to detect riders and check for helmet presence. Riders without helmets in the head region are flagged as violations.

### Wrong Lane Detection
- Define prohibited zones using a binary mask (white = forbidden, black = allowed)
- Calculate vehicle centroid
- Flag violation if centroid falls in white zone

### License Plate Recognition
1. Detect license plates using YOLOv8
2. Crop and preprocess the plate region
3. Apply Tesseract OCR with optimized configuration
4. Normalize and validate extracted text

## Configuration

### Wrong Lane Mask
Create a binary mask at `data/masks/wrong_lane_mask.png`:
- White (255) = Forbidden zone
- Black (0) = Allowed zone

### Model Weights
Specify custom weights:
```bash
python app/main.py --image sample.jpg \
  --helmet-weights models/custom_helmet.pt \
  --plate-weights models/custom_plate.pt
```

## Training Custom Models

### License Plate Detector

1. Prepare dataset in YOLO format:
```
datasets/plates/
  images/train/
  images/val/
  labels/train/
  labels/val/
plate.yaml
```

2. Train the model:
```bash
yolo detect train model=yolov8n.pt data=plate.yaml epochs=100 batch=16
```

3. Copy trained weights:
```bash
cp runs/detect/train/weights/best.pt models/lp_yolov8.pt
```

For detailed training instructions, see the documentation.

## Performance

- **CPU**: ~2-5 FPS on modern processors
- **GPU**: ~30-60 FPS on NVIDIA RTX 3060+
- **Apple Silicon**: ~10-15 FPS with MPS acceleration

## API Reference

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--image` | Path to single image | - |
| `--folder` | Path to image folder | - |
| `--plates` | Enable plate detection | False |
| `--helmet-weights` | Path to helmet model | models/helmet_yolov8.pt |
| `--plate-weights` | Path to plate model | models/lp_yolov8.pt |

## Troubleshooting

### Common Issues

**Tesseract not found:**
```python
# Set Tesseract path manually (Windows)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**GPU not detected:**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Poor OCR accuracy:**
- Improve image quality and lighting
- Try `--psm 7` for single-line plates
- Adjust preprocessing parameters in `app/plates.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition
- [OpenCV](https://opencv.org/) for image processing
- Kaggle community for helmet detection datasets

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Built for safer roads through AI-powered traffic monitoring**
