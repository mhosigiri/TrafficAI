#!/usr/bin/env python3
"""
GPU Setup Verification Script
Check if your system is ready for GPU-accelerated training.
"""

import sys
import torch
from ultralytics import YOLO

def check_pytorch_gpu():
    """Check PyTorch GPU setup."""
    print("🔍 PyTorch GPU Setup Check")
    print("=" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            memory_gb = gpu.total_memory / 1024**3
            print(f"  GPU {i}: {gpu.name}")
            print(f"    Memory: {memory_gb:.1f} GB")
            print(f"    Compute capability: {gpu.major}.{gpu.minor}")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        
    print()

def check_ultralytics_gpu():
    """Check Ultralytics YOLO GPU support."""
    print("🔍 Ultralytics YOLO GPU Check")
    print("=" * 40)
    
    try:
        # Create a simple model to test GPU
        model = YOLO('yolov8n.pt')
        
        if torch.cuda.is_available():
            print("Testing CUDA support...")
            model.to('cuda')
            print("✅ CUDA support working")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Testing MPS support...")
            model.to('mps')
            print("✅ MPS support working")
        else:
            print("⚠️ No GPU acceleration available - will use CPU")
            
    except Exception as e:
        print(f"❌ Error testing GPU support: {e}")
    
    print()

def get_recommendations():
    """Provide optimization recommendations."""
    print("💡 Training Recommendations")
    print("=" * 40)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 8:
            print("✅ Excellent! You can use:")
            print("   - Batch size: 16-32")
            print("   - Image size: 640 or higher")
            print("   - Multiple workers: 8+")
        elif gpu_memory >= 4:
            print("✅ Good! Recommended settings:")
            print("   - Batch size: 8-16")
            print("   - Image size: 640")
            print("   - Workers: 4-8")
        else:
            print("⚠️ Limited GPU memory. Use:")
            print("   - Batch size: 4-8")
            print("   - Image size: 640")
            print("   - Workers: 4")
            
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple Silicon detected:")
        print("   - Batch size: 4-8")
        print("   - Image size: 640")
        print("   - Workers: 4-6")
        
    else:
        print("💻 CPU training (slower):")
        print("   - Batch size: 2-4")
        print("   - Image size: 640")
        print("   - Workers: 2-4")
        print("   - Consider reducing epochs")

def main():
    """Main GPU check function."""
    print("🚀 TrafficAI GPU Setup Verification")
    print("=" * 50)
    print()
    
    check_pytorch_gpu()
    check_ultralytics_gpu()
    get_recommendations()
    
    print("🎯 Ready to train with optimized settings!")
    print("Run: python train_model.py")

if __name__ == "__main__":
    main()
