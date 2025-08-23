#!/usr/bin/env python3
"""
TrafficAI - GPU Optimized Training Script
Force GPU training with optimal settings for RTX 4060.
"""

import os
import torch
from ultralytics import YOLO

def main():
    """Train helmet detection model with GPU optimization."""
    
    print("🚀 TrafficAI - GPU Optimized Training")
    print("=" * 50)
    
    # Verify GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please check your PyTorch installation.")
        return
    
    # GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🚀 GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_memory:.1f} GB")
    
    # Optimal settings for RTX 4060 (8GB)
    device = "0"  # Use GPU 0
    batch_size = 16  # Optimal for 8GB GPU
    epochs = 100  # More epochs with GPU
    workers = 8
    
    print(f"\n🏋️ Training Configuration:")
    print(f"   - Device: GPU {device}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Workers: {workers}")
    print(f"   - Mixed Precision: Enabled")
    print(f"   - Image Cache: Enabled")
    
    # Check dataset
    dataset_config = "config/dataset.yaml"
    if not os.path.exists(dataset_config):
        print(f"❌ Dataset config not found: {dataset_config}")
        return
    
    # Initialize model
    print(f"\n🔧 Initializing YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Start training with GPU optimization
    print(f"\n🚀 Starting GPU training...")
    try:
        results = model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            name='helmet_detection_gpu',
            patience=20,  # More patience for better results
            save=True,
            device=device,  # Force GPU 0
            verbose=True,
            amp=True,      # Automatic Mixed Precision
            cache=True,    # Cache images in GPU memory
            workers=workers,
            # Additional optimizations
            optimizer='AdamW',  # Better optimizer
            lr0=0.001,     # Lower learning rate for stability
            cos_lr=True,   # Cosine learning rate scheduling
            close_mosaic=20,  # Disable mosaic in last 20 epochs
        )
        
        # Copy best model
        best_model_path = "runs/detect/helmet_detection_gpu/weights/best.pt"
        target_model_path = "models/helmet_yolov8_gpu.pt"
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, target_model_path)
            print(f"\n✅ GPU Training completed successfully!")
            print(f"📦 Best model saved to: {target_model_path}")
            print(f"📊 Training results: runs/detect/helmet_detection_gpu/")
            
        else:
            print(f"❌ Training completed but model not found at: {best_model_path}")
            
    except Exception as e:
        print(f"❌ GPU Training failed: {e}")
        print("Consider reducing batch size if out of memory.")

if __name__ == "__main__":
    main()
