#!/usr/bin/env python3
"""
TrafficAI Setup Script
Automatically configures TrafficAI for any user on any system.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_banner():
    """Print the setup banner."""
    print("=" * 60)
    print("🚦 TrafficAI Setup Script")
    print("=" * 60)
    print("This script will automatically set up TrafficAI on your system.")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def detect_system():
    """Detect the operating system and architecture."""
    system = platform.system()
    machine = platform.machine()
    print(f"🖥️  System: {system} {machine}")
    return system, machine

def install_pytorch(system, machine):
    """Install PyTorch based on system and architecture."""
    print("\n🔧 Installing PyTorch...")
    
    if system == "Darwin" and machine == "arm64":  # Apple Silicon
        print("🍎 Detected Apple Silicon Mac - installing MPS-compatible PyTorch")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    elif system == "Darwin":  # Intel Mac
        print("🍎 Detected Intel Mac - installing CPU PyTorch")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    elif system == "Windows":
        print("🪟 Detected Windows - installing CPU PyTorch")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    elif system == "Linux":
        print("🐧 Detected Linux - installing CPU PyTorch")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        print("❓ Unknown system - installing CPU PyTorch")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])

def create_virtual_environment():
    """Create a virtual environment."""
    print("\n🔧 Creating virtual environment...")
    
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✅ Virtual environment created successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        print("Please install python3-venv package:")
        if platform.system() == "Ubuntu" or platform.system() == "Debian":
            print("sudo apt-get install python3-venv")
        elif platform.system() == "Darwin":
            print("brew install python3")
        sys.exit(1)

def activate_and_install():
    """Activate virtual environment and install dependencies."""
    print("\n🔧 Installing dependencies...")
    
    # Determine activation script based on OS
    if platform.system() == "Windows":
        activate_script = ".venv\\Scripts\\activate.bat"
        pip_path = ".venv\\Scripts\\pip.exe"
    else:
        activate_script = ".venv/bin/activate"
        pip_path = ".venv/bin/pip"
    
    # Install requirements
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_path, "install", "-r", "web_requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def check_models():
    """Check if required models exist."""
    print("\n🔍 Checking AI models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠️  Models directory not found")
        print("Please ensure you have the following models:")
        print("  - models/helmet_yolov8.pt (or helmet_yolov8_gpu.pt)")
        print("  - models/license_plate_yolov8.pt")
        return False
    
    helmet_model = models_dir / "helmet_yolov8.pt"
    helmet_gpu_model = models_dir / "helmet_yolov8_gpu.pt"
    license_model = models_dir / "license_plate_yolov8.pt"
    
    if not (helmet_model.exists() or helmet_gpu_model.exists()):
        print("⚠️  Helmet detection model not found")
        print("Please place helmet_yolov8.pt in the models/ directory")
        return False
    
    if not license_model.exists():
        print("⚠️  License plate detection model not found")
        print("Please place license_plate_yolov8.pt in the models/ directory")
        return False
    
    print("✅ All required models found")
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["uploads", "results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/ directory")

def create_startup_script():
    """Create a startup script for easy launching."""
    print("\n🚀 Creating startup script...")
    
    if platform.system() == "Windows":
        script_content = """@echo off
echo Starting TrafficAI Web Application...
call .venv\\Scripts\\activate.bat
python web_app.py
pause
"""
        script_path = "start_trafficai.bat"
    else:
        script_content = """#!/bin/bash
echo "Starting TrafficAI Web Application..."
source .venv/bin/activate
python web_app.py
"""
        script_path = "start_trafficai.sh"
        # Make executable
        os.chmod(script_path, 0o755)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"✅ Created {script_path}")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Ensure you have the AI models in the models/ directory")
    print("2. Run the startup script:")
    
    if platform.system() == "Windows":
        print("   start_trafficai.bat")
    else:
        print("   ./start_trafficai.sh")
    
    print("\n3. Open your browser and go to: http://localhost:3000")
    print("\n📚 For more information, see README_WEB.md")
    print("=" * 60)

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Detect system
    system, machine = detect_system()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install PyTorch first
    install_pytorch(system, machine)
    
    # Install other dependencies
    activate_and_install()
    
    # Check models
    check_models()
    
    # Create directories
    create_directories()
    
    # Create startup script
    create_startup_script()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
