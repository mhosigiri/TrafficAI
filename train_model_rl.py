#!/usr/bin/env python3
"""
TrafficAI - Enhanced Model Training Script with Reinforcement Learning
Train helmet detection models with initial dataset + continuous RL from user feedback.
"""

import os
import sys
import torch
from pathlib import Path
import argparse

# Import our RL training components
from app.rl_trainer import RLHelmetTrainer


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
        return device, True, None
    else:
        device = "cpu"
        print("💻 Using CPU (consider installing CUDA for GPU acceleration)")
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


def train_base_model(dataset_config="config/dataset.yaml", epochs=None):
    """Train the base model using the original dataset.
    
    Args:
        dataset_config: Path to dataset configuration
        epochs: Number of epochs (auto-determined if None)
    """
    try:
        from ultralytics import YOLO
        
        # Check device availability
        device, has_gpu, gpu_memory = check_device()
        
        # Get optimal batch size
        batch_size = get_optimal_batch_size(has_gpu, gpu_memory)
        
        # Adjust epochs based on device
        if epochs is None:
            epochs = 100 if has_gpu else 50
        
        print(f"\n🏋️ Starting base model training...")
        print(f"📊 Training parameters:")
        print(f"   - Dataset: {dataset_config}")
        print(f"   - Model: YOLOv8 Nano")
        print(f"   - Epochs: {epochs}")
        print(f"   - Image size: 640")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Device: {device.upper()}")
        print(f"   - Hardware acceleration: {'✅' if has_gpu else '❌'}")
        
        # Check if dataset configuration exists
        if not os.path.exists(dataset_config):
            print(f"❌ Dataset configuration not found: {dataset_config}")
            return False
        
        # Check if we have training data
        images_dir = "data/images"
        labels_dir = "data/labels"
        
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
            
        if not os.path.exists(labels_dir):
            print(f"❌ Labels directory not found: {labels_dir}")
            return False
        
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
            return False
            
        if label_files == 0:
            print("❌ No label files found in data/labels/")
            return False
        
        # Initialize model
        print(f"\n🔧 Initializing YOLOv8 model...")
        model = YOLO('yolov8n.pt')  # Start with nano model
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Train the model
        results = model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            name='helmet_detection_rl_base',
            patience=15 if has_gpu else 10,
            save=True,
            device=device,
            verbose=True,
            # Additional GPU optimizations
            amp=has_gpu,  # Automatic Mixed Precision for faster training on GPU
            cache=has_gpu,  # Cache images in memory when using GPU
            workers=8 if has_gpu else 4  # More workers for data loading with GPU
        )
        
        # Copy best model to models directory
        best_model_path = "runs/detect/helmet_detection_rl_base/weights/best.pt"
        target_model_path = "models/helmet_yolov8.pt"
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, target_model_path)
            print(f"\n✅ Base training completed successfully!")
            print(f"📦 Best model saved to: {target_model_path}")
            print(f"📊 Training results saved in: runs/detect/helmet_detection_rl_base/")
            return True
        else:
            print(f"❌ Training completed but best model not found at: {best_model_path}")
            return False
            
    except ImportError:
        print("❌ Ultralytics not installed. Please run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Base training failed: {e}")
        return False


def setup_rl_system():
    """Setup the reinforcement learning system."""
    try:
        print(f"\n🤖 Setting up Reinforcement Learning system...")
        
        # Initialize RL trainer
        rl_trainer = RLHelmetTrainer()
        
        # Get current stats
        stats = rl_trainer.get_training_stats()
        
        print(f"✅ RL system initialized successfully!")
        print(f"📊 Current RL statistics:")
        print(f"   - Total feedback: {stats['total_feedback']}")
        print(f"   - Processed feedback: {stats['processed_feedback']}")
        print(f"   - Pending feedback: {stats['pending_feedback']}")
        print(f"   - Replay buffer size: {stats['replay_buffer_size']}")
        
        return rl_trainer
        
    except Exception as e:
        print(f"❌ Failed to setup RL system: {e}")
        return None


def train_with_rl_feedback(rl_trainer, epochs=10):
    """Train the model using collected user feedback.
    
    Args:
        rl_trainer: RLHelmetTrainer instance
        epochs: Number of RL training epochs
    """
    if not rl_trainer:
        print("❌ RL trainer not available")
        return False
    
    try:
        print(f"\n🧠 Starting Reinforcement Learning training...")
        
        # Check available feedback
        stats = rl_trainer.get_training_stats()
        
        if stats['pending_feedback'] == 0:
            print(f"⚠️ No user feedback available for RL training")
            print(f"💡 Use the feedback interface to collect user corrections:")
            print(f"   python -m app.main_rl --gui")
            return False
        
        print(f"📝 Training with {stats['pending_feedback']} feedback examples...")
        
        # Train from feedback
        rl_trainer.train_from_feedback(epochs=epochs, min_feedback_count=1)
        
        print(f"✅ RL training completed!")
        return True
        
    except Exception as e:
        print(f"❌ RL training failed: {e}")
        return False


def main():
    """Main training function with RL support."""
    parser = argparse.ArgumentParser(description="TrafficAI Enhanced Training with Reinforcement Learning")
    parser.add_argument("--base-only", action="store_true", help="Only train base model (skip RL)")
    parser.add_argument("--rl-only", action="store_true", help="Only train with RL feedback (skip base)")
    parser.add_argument("--epochs", type=int, help="Number of epochs for base training")
    parser.add_argument("--rl-epochs", type=int, default=10, help="Number of epochs for RL training")
    parser.add_argument("--dataset", type=str, default="config/dataset.yaml", help="Dataset configuration path")
    
    args = parser.parse_args()
    
    print("🚀 TrafficAI Enhanced Training with Reinforcement Learning")
    print("=" * 65)
    
    success = True
    
    # Base model training
    if not args.rl_only:
        print(f"\n📚 Phase 1: Base Model Training")
        print("-" * 40)
        success &= train_base_model(args.dataset, args.epochs)
        
        if not success:
            print(f"❌ Base training failed. Stopping.")
            return
    
    # RL system setup
    if not args.base_only:
        print(f"\n🤖 Phase 2: Reinforcement Learning Setup")
        print("-" * 45)
        rl_trainer = setup_rl_system()
        
        if rl_trainer:
            # RL training (if feedback available)
            print(f"\n🧠 Phase 3: RL Training from User Feedback")
            print("-" * 50)
            rl_success = train_with_rl_feedback(rl_trainer, args.rl_epochs)
            
            if not rl_success:
                print(f"💡 RL training skipped - collect feedback first!")
        else:
            print(f"❌ RL setup failed")
            success = False
    
    # Final summary
    print(f"\n🎯 Training Summary")
    print("=" * 20)
    
    if success:
        print(f"✅ Training pipeline completed successfully!")
        print(f"\n📋 Next Steps:")
        print(f"1. 🔍 Test detection: python -m app.main_rl --image data/images/sample.jpg")
        print(f"2. 📝 Collect feedback: python -m app.main_rl --gui")
        print(f"3. 🏋️ RL training: python train_model_rl.py --rl-only")
        print(f"4. 📊 Check stats: python -m app.main_rl --stats")
        
        print(f"\n🔄 Continuous Learning Workflow:")
        print(f"   Users provide feedback → Model improves → Better detection!")
        
    else:
        print(f"❌ Training pipeline completed with errors")
        print(f"💡 Check error messages above and try again")


if __name__ == "__main__":
    main()
