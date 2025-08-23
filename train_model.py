#!/usr/bin/env python3
"""
TrafficAI - Model Training Script
Script to train a YOLOv8 model for helmet detection using your dataset.
"""

import os
import sys
from pathlib import Path
import torch

def check_device():
    """Check for GPU availability and return the best device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        return device, True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Apple Silicon GPU (MPS) Available")
        return device, True
    else:
        device = "cpu"
        print("💻 Using CPU (consider installing CUDA for GPU acceleration)")
        return device, False

def get_optimal_batch_size(has_gpu, gpu_memory=None):
    """Get optimal batch size based on available hardware."""
    if has_gpu:
        if gpu_memory and gpu_memory >= 8:
            return 16  # High-end GPU
        elif gpu_memory and gpu_memory >= 4:
            return 8   # Mid-range GPU
        else:
            return 4   # Entry-level GPU or MPS
    else:
        return 2       # CPU fallback

def main():
    """Train a helmet detection model."""
    
    print("🚀 TrafficAI - Helmet Detection Model Training")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # Check device availability
        device, has_gpu = check_device()
        gpu_memory = None
        if has_gpu and device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Get optimal batch size
        batch_size = get_optimal_batch_size(has_gpu, gpu_memory)
        
        # Ensure we use a good batch size for the RTX 4060
        if has_gpu and gpu_memory and gpu_memory >= 8:
            batch_size = 16  # Optimal for 8GB GPU
        
        # Check if dataset configuration exists
        dataset_config = "config/dataset.yaml"
        if not os.path.exists(dataset_config):
            print(f"❌ Dataset configuration not found: {dataset_config}")
            print("Please ensure you have:")
            print("1. Images in data/images/")
            print("2. Labels in data/labels/ (YOLO format)")
            print("3. Dataset config at config/dataset.yaml")
            return
        
        # Check if we have training data
        images_dir = "data/images"
        labels_dir = "data/labels"
        
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return
            
        if not os.path.exists(labels_dir):
            print(f"❌ Labels directory not found: {labels_dir}")
            return
        
        # Count files
        image_files = len([f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        label_files = len([f for f in os.listdir(labels_dir) 
                          if f.lower().endswith('.txt')])
        
        print(f"📊 Dataset Overview:")
        print(f"   - Images: {image_files}")
        print(f"   - Labels: {label_files}")
        
        if image_files == 0:
            print("❌ No images found in data/images/")
            return
            
        if label_files == 0:
            print("❌ No label files found in data/labels/")
            print("Use app/xml_to_yolo_converter.py to convert XML annotations")
            return
        
        # Initialize model
        print("\n🔧 Initializing YOLOv8 model...")
        model = YOLO('yolov8n.pt')  # Start with nano model
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Adjust epochs based on device
        epochs = 100 if has_gpu else 50
        
        # Force GPU device if available
        if has_gpu and device == "cuda":
            device = "0"  # Use GPU 0 explicitly
        
        # Training parameters  
        print(f"\n🏋️ Starting training with dataset: {dataset_config}")
        print("Training parameters:")
        print("   - Model: YOLOv8 Nano")
        print(f"   - Epochs: {epochs}")
        print("   - Image size: 640")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Device: {device.upper()}")
        print(f"   - Hardware acceleration: {'✅' if has_gpu else '❌'}")
        
        # Train the model
        print(f"🔧 Using device: {device}")
        print(f"🔧 Batch size: {batch_size}")
        print(f"🔧 GPU acceleration: {'✅' if has_gpu else '❌'}")
        
        results = model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            name='helmet_detection',
            patience=15 if has_gpu else 10,  # More patience with GPU
            save=True,
            device=device,
            verbose=True,
            # Additional GPU optimizations
            amp=has_gpu,  # Automatic Mixed Precision for faster training on GPU
            cache=has_gpu,  # Cache images in memory when using GPU
            workers=8 if has_gpu else 4  # More workers for data loading with GPU
        )
        
        # Copy best model to models directory
        best_model_path = "runs/detect/helmet_detection/weights/best.pt"
        target_model_path = "models/helmet_yolov8.pt"
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, target_model_path)
            print(f"\n✅ Training completed successfully!")
            print(f"📦 Best model copied to: {target_model_path}")
            print(f"📊 Training results saved in: runs/detect/helmet_detection/")
            
            # Print next steps
            print(f"\n🎯 Next Steps:")
            print(f"1. Test your model: python app/main.py --image data/samples/test_image.jpg")
            print(f"2. Process a folder: python app/main.py --folder data/samples")
            print(f"3. Check training results in: runs/detect/helmet_detection/")
            
        else:
            print(f"❌ Training completed but best model not found at: {best_model_path}")
            
    except ImportError:
        print("❌ Ultralytics not installed. Please run: pip install ultralytics")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        

if __name__ == "__main__":
    main()
