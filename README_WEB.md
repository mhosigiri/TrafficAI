# 🚦 TrafficAI Web Application

A modern, interactive web interface for TrafficAI that detects helmet violations and license plates using computer vision and AI.

## ✨ Features

- **🪖 Helmet Detection**: Automatically detects riders with and without helmets
- **🚗 License Plate Recognition**: Extracts and identifies license plates
- **📹 Video Analysis**: Process videos frame-by-frame with timeline visualization
- **🖼️ Image Processing**: Upload and analyze individual images
- **🎨 Modern UI**: Beautiful, responsive web interface with drag-and-drop
- **📊 Real-time Results**: Instant detection results with bounding boxes
- **🔒 Safety Status**: Clear indicators for safe/unsafe riding

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/mhosigiri/TrafficAI.git
   cd TrafficAI
   ```

2. **Run the setup script**
   ```bash
   python setup_trafficai.py
   ```

3. **Start the application**
   - **Windows**: Double-click `start_trafficai.bat`
   - **Mac/Linux**: Run `./start_trafficai.sh`

4. **Open your browser**
   Navigate to: http://localhost:3000

### Option 2: Manual Setup

1. **Install Python 3.8+**
   - [Download Python](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked during installation

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   - **Windows**: `.venv\Scripts\activate`
   - **Mac/Linux**: `source .venv/bin/activate`

4. **Install dependencies**
   ```bash
   pip install -r web_requirements.txt
   ```

5. **Run the application**
   ```bash
   python web_app.py
   ```

## 📋 Requirements

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space

### AI Models Required
Place these files in the `models/` directory:
- `helmet_yolov8.pt` - Helmet detection model
- `license_plate_yolov8.pt` - License plate detection model

## 🖥️ Supported Platforms

| Platform | Status | Notes |
|----------|---------|-------|
| **Windows 10/11** | ✅ Full Support | CPU inference |
| **macOS Intel** | ✅ Full Support | CPU inference |
| **macOS Apple Silicon** | ✅ Full Support | MPS acceleration |
| **Ubuntu 20.04+** | ✅ Full Support | CPU inference |
| **CentOS 7+** | ✅ Full Support | CPU inference |

## 🔧 Troubleshooting

### Common Issues

#### 1. "Python not found"
- Ensure Python is installed and added to PATH
- Try `python3` instead of `python`

#### 2. "Permission denied" (Linux/Mac)
- Make setup script executable: `chmod +x setup_trafficai.py`
- Use `sudo` if needed for system packages

#### 3. "Module not found" errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r web_requirements.txt`

#### 4. "Port 3000 already in use"
- Change port in `web_app.py` (line 432)
- Or kill existing process: `lsof -ti:3000 | xargs kill -9`

#### 5. "Models not found"
- Check `models/` directory exists
- Ensure model files have correct names
- Download models from the original TrafficAI repository

### Performance Tips

- **GPU Users**: Install CUDA-compatible PyTorch for faster inference
- **Apple Silicon**: MPS acceleration is automatically enabled
- **Memory Issues**: Reduce video frame processing limit in `web_app.py`

## 📱 Usage Guide

### Uploading Files
1. **Drag & Drop**: Simply drag files onto the upload area
2. **Click to Browse**: Click the upload area to select files
3. **Supported Formats**:
   - Images: JPG, PNG, GIF
   - Videos: MP4, AVI, MOV, WMV, FLV, WEBM

### Processing Options
- **Helmet Detection**: Enable/disable helmet detection
- **License Plate Detection**: Enable/disable license plate recognition
- **Both**: Run both detections simultaneously

### Viewing Results
- **Images**: Annotated image with bounding boxes
- **Videos**: Timeline of frames with detection results
- **Statistics**: Count of safe/unsafe riders and license plates

## 🏗️ Architecture

```
TrafficAI Web App/
├── web_app.py              # Flask backend server
├── app/                    # Core AI modules
│   ├── helmet_detector.py  # Helmet detection logic
│   └── license_plate_detector.py  # License plate detection
├── templates/              # HTML templates
│   └── index.html         # Main web interface
├── static/                 # Frontend assets
│   ├── css/style.css      # Styling
│   └── js/app.js          # JavaScript functionality
├── models/                 # AI model files
├── uploads/                # Temporary file storage
└── results/                # Processed results
```

## 🔒 Security Features

- File type validation
- File size limits (500MB max)
- Secure filename handling
- Temporary file cleanup
- No persistent file storage

## 🌐 Deployment

### Local Development
```bash
python web_app.py
```

### Production Deployment
```bash
# Install production WSGI server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:3000 web_app:app
```

### Docker Deployment
```bash
# Build image
docker build -t trafficai-web .

# Run container
docker run -p 3000:3000 trafficai-web
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original TrafficAI project by [mhosigiri](https://github.com/mhosigiri)
- YOLOv8 by Ultralytics
- Flask web framework
- OpenCV for computer vision

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mhosigiri/TrafficAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mhosigiri/TrafficAI/discussions)
- **Documentation**: [README_WEB.md](README_WEB.md)

---

**Made with ❤️ for safer roads**
