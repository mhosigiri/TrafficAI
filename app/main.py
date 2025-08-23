#!/usr/bin/env python3
"""
TrafficAI - Helmet Detection Main Application
A simple traffic safety monitoring system using YOLOv8 for helmet detection.
"""

import argparse
import os
import glob
from pathlib import Path
import sys

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helmet_detector import HelmetDetector


def process_single_image(detector, image_path):
    """Process a single image for helmet detection.
    
    Args:
        detector: HelmetDetector instance
        image_path: Path to the image file
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    result = detector.process_image(image_path, save_result=True)
    
    # Print violation summary
    if result['no_helmet_count'] > 0:
        print(f"⚠️ VIOLATION: {result['no_helmet_count']} rider(s) without helmet detected!")
    else:
        print("✅ No helmet violations detected")


def process_folder(detector, folder_path):
    """Process all images in a folder.
    
    Args:
        detector: HelmetDetector instance
        folder_path: Path to folder containing images
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"📁 Processing {len(image_files)} images from {folder_path}")
    
    total_violations = 0
    processed_count = 0
    
    for image_path in sorted(image_files):
        print(f"\n{'='*60}")
        result = detector.process_image(image_path, save_result=True)
        
        if result['no_helmet_count'] > 0:
            total_violations += result['no_helmet_count']
            print(f"⚠️ VIOLATION: {result['no_helmet_count']} rider(s) without helmet!")
        
        processed_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {processed_count}")
    print(f"Total helmet violations: {total_violations}")
    
    if total_violations > 0:
        print(f"⚠️ Action required: {total_violations} helmet violations detected!")
    else:
        print("✅ All clear - no helmet violations detected")


def train_model(dataset_yaml_path):
    """Train a new YOLOv8 model for helmet detection.
    
    Args:
        dataset_yaml_path: Path to the dataset configuration file
    """
    if not os.path.exists(dataset_yaml_path):
        print(f"❌ Dataset configuration not found: {dataset_yaml_path}")
        return
    
    try:
        from ultralytics import YOLO
        
        print("🚀 Starting YOLOv8 training...")
        print(f"📄 Using dataset config: {dataset_yaml_path}")
        
        # Initialize a YOLOv8 nano model
        model = YOLO('yolov8n.pt')  # Start with pretrained weights
        
        # Train the model
        results = model.train(
            data=dataset_yaml_path,
            epochs=100,
            imgsz=640,
            batch=16,
            name='helmet_detection',
            patience=10,
            save=True,
            device='cpu'  # Change to 'cuda' if you have a GPU
        )
        
        print("✅ Training completed!")
        print(f"Best model saved at: runs/detect/helmet_detection/weights/best.pt")
        print("Copy the best.pt file to models/helmet_yolov8.pt to use it for inference")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="TrafficAI - Helmet Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image data/samples/sample1.jpg
  %(prog)s --folder data/samples
  %(prog)s --train config/dataset.yaml
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to a single image to process')
    parser.add_argument('--folder', type=str, help='Path to folder containing images to process')
    parser.add_argument('--train', type=str, help='Train model using dataset.yaml file')
    parser.add_argument('--model', type=str, default='models/helmet_yolov8.pt', 
                       help='Path to custom model (default: models/helmet_yolov8.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if not any([args.image, args.folder, args.train]):
        parser.print_help()
        return
    
    # Training mode
    if args.train:
        train_model(args.train)
        return
    
    # Initialize detector for inference
    print("🔧 Initializing TrafficAI Helmet Detection System...")
    detector = HelmetDetector(model_path=args.model, confidence=args.confidence)
    
    # Process single image
    if args.image:
        process_single_image(detector, args.image)
    
    # Process folder
    elif args.folder:
        process_folder(detector, args.folder)


if __name__ == "__main__":
    main()
