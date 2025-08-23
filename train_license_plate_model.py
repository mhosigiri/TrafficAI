#!/usr/bin/env python3
"""
License Plate Detection Model Training Script
Train a YOLOv8 model specifically for license plate detection.
"""

import os
import sys
from pathlib import Path
import torch
import requests
import zipfile
from ultralytics import YOLO

def download_license_plate_dataset():
    """
    Download and prepare license plate dataset.
    This function sets up a sample dataset structure for license plate training.
    """
    
    print("📥 Setting up license plate dataset...")
    
    # Create directory structure
    base_dir = Path("data/license_plates")
    os.makedirs(base_dir / "images", exist_ok=True)
    os.makedirs(base_dir / "labels", exist_ok=True)
    os.makedirs(base_dir / "annotations", exist_ok=True)
    
    # Create sample dataset configuration
    dataset_config = """# License Plate Detection Dataset
train: data/license_plates/images
val: data/license_plates/images
test: data/license_plates/images

nc: 1
names:
  0: license_plate
"""
    
    config_path = base_dir / "dataset.yaml"
    with open(config_path, 'w') as f:
        f.write(dataset_config)
    
    print(f"✅ Dataset structure created at: {base_dir}")
    print(f"📝 Dataset config saved to: {config_path}")
    
    # Instructions for user
    print("\n📋 Next Steps:")
    print("1. Download license plate dataset from the GitHub repository:")
    print("   https://github.com/Asikpalysik/Automatic-License-Plate-Detection")
    print("2. Extract images to: data/license_plates/source_images/")
    print("3. Extract XML annotations to: data/license_plates/annotations/")
    print("4. Run: python app/xml_to_yolo_license_plates.py")
    print("5. Then run this training script again")
    
    return config_path

def check_device():
    """Check for GPU availability and return the best device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        return device, True, gpu_memory
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Apple Silicon GPU (MPS) Available")
        return device, True, 8.0  # Assume 8GB for MPS
    else:
        device = "cpu"
        print("💻 Using CPU (consider using GPU for faster training)")
        return device, False, None

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

def train_license_plate_model():
    """Train license plate detection model."""
    
    print("🚀 License Plate Detection Model Training")
    print("=" * 50)
    
    # Check device availability
    device, has_gpu, gpu_memory = check_device()
    
    # Get optimal batch size
    batch_size = get_optimal_batch_size(has_gpu, gpu_memory)
    
    # Check if dataset exists
    dataset_config = "data/license_plates/dataset.yaml"
    
    if not os.path.exists(dataset_config):
        print("❌ License plate dataset not found!")
        config_path = download_license_plate_dataset()
        print("\n⚠️  Please follow the instructions above to set up the dataset.")
        return
    
    # Check if we have training data
    images_dir = "data/license_plates/images"
    labels_dir = "data/license_plates/labels"
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("❌ Dataset directories not found!")
        print("Please run: python app/xml_to_yolo_license_plates.py")
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
        print("❌ No images found in dataset!")
        return
    
    if label_files == 0:
        print("❌ No label files found!")
        print("Please run: python app/xml_to_yolo_license_plates.py")
        return
    
    # Initialize model
    print("\n🔧 Initializing YOLOv8 model for license plate detection...")
    model = YOLO('yolov8n.pt')  # Start with nano model
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Adjust epochs based on device and dataset size
    if image_files > 100:
        epochs = 100 if has_gpu else 50
    else:
        epochs = 50 if has_gpu else 25
    
    # Training parameters  
    print(f"\n🏋️ Starting license plate training...")
    print("Training parameters:")
    print("   - Model: YOLOv8 Nano (License Plate Detection)")
    print(f"   - Epochs: {epochs}")
    print("   - Image size: 640")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Device: {device.upper()}")
    print(f"   - Hardware acceleration: {'✅' if has_gpu else '❌'}")
    
    try:
        # Train the model
        results = model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            name='license_plate_detection',
            patience=20 if has_gpu else 15,
            save=True,
            device=device,
            verbose=True,
            # Optimizations for license plate detection
            amp=has_gpu,  # Automatic Mixed Precision
            cache=has_gpu,  # Cache images in memory
            workers=8 if has_gpu else 4,
            # License plate specific optimizations
            cos_lr=True,   # Cosine learning rate scheduling
            close_mosaic=20,  # Disable mosaic in last 20 epochs
            optimizer='AdamW',  # Better optimizer for small objects
            lr0=0.001,     # Lower learning rate for stability
        )
        
        # Copy best model to models directory
        best_model_path = "runs/detect/license_plate_detection/weights/best.pt"
        target_model_path = "models/license_plate_yolov8.pt"
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, target_model_path)
            print(f"\n✅ License plate training completed successfully!")
            print(f"📦 Best model copied to: {target_model_path}")
            print(f"📊 Training results saved in: runs/detect/license_plate_detection/")
            
            # Print next steps
            print(f"\n🎯 Next Steps:")
            print(f"1. Test license plate detection: python app/license_plate_detector.py")
            print(f"2. Use integrated system: python app/integrated_traffic_monitor.py")
            print(f"3. Check training results in: runs/detect/license_plate_detection/")
            
            # Test the trained model
            test_trained_model(target_model_path)
            
        else:
            print(f"❌ Training completed but best model not found at: {best_model_path}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This might be due to insufficient dataset or configuration issues.")

def test_trained_model(model_path):
    """Test the trained license plate model."""
    print(f"\n🔬 Testing trained model: {model_path}")
    
    try:
        from app.license_plate_detector import LicensePlateDetector
        
        # Initialize detector with trained model
        detector = LicensePlateDetector(model_path=model_path)
        
        # Test on sample images if available
        test_images = [
            "data/license_plates/images",
            "data/images"
        ]
        
        for test_dir in test_images:
            if os.path.exists(test_dir):
                image_files = [f for f in os.listdir(test_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    test_image_path = os.path.join(test_dir, image_files[0])
                    print(f"🖼️  Testing on: {test_image_path}")
                    
                    import cv2
                    image = cv2.imread(test_image_path)
                    if image is not None:
                        annotated, detections = detector.detect_plates(image)
                        print(f"   Found {len(detections)} license plate(s)")
                        
                        for i, det in enumerate(detections):
                            print(f"     Plate {i+1}: confidence={det['confidence']:.3f}")
                    break
        
        print("✅ Model test completed successfully!")
        
    except ImportError as e:
        print(f"⚠️  Could not test model: {e}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def main():
    """Main training function."""
    try:
        train_license_plate_model()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
