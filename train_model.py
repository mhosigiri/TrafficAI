#!/usr/bin/env python3
"""
TrafficAI - Model Training Script
Script to train a YOLOv8 model for helmet detection using your dataset.
"""

import os
import sys
from pathlib import Path

def main():
    """Train a helmet detection model."""
    
    print("🚀 TrafficAI - Helmet Detection Model Training")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
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
        
        # Training parameters  
        print(f"\n🏋️ Starting training with dataset: {dataset_config}")
        print("Training parameters:")
        print("   - Model: YOLOv8 Nano")
        print("   - Epochs: 50 (reduced for quick demo)")
        print("   - Image size: 640")
        print("   - Batch size: 4 (reduced for CPU)")
        print("   - Device: CPU")
        
        # Train the model
        results = model.train(
            data=dataset_config,
            epochs=50,  # Reduced for faster training on CPU
            imgsz=640,
            batch=4,    # Smaller batch for CPU
            name='helmet_detection',
            patience=10,
            save=True,
            device='cpu',  # Force CPU for Mac compatibility
            verbose=True
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
