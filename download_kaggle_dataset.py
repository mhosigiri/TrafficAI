#!/usr/bin/env python3
"""
TrafficAI - Kaggle Dataset Downloader
Automatically downloads the helmet detection dataset from Kaggle using kagglehub.
"""

import os
import shutil
import glob
from pathlib import Path


def download_helmet_dataset():
    """Download the helmet detection dataset from Kaggle."""
    print("🚀 TrafficAI - Kaggle Dataset Downloader")
    print("=" * 50)
    
    try:
        import kagglehub
        print("✅ kagglehub imported successfully")
    except ImportError:
        print("❌ kagglehub not found. Installing...")
        import subprocess
        subprocess.check_call(["pip3", "install", "kagglehub"])
        import kagglehub
        print("✅ kagglehub installed and imported")
    
    try:
        print("\n📥 Downloading helmet detection dataset from Kaggle...")
        print("Dataset: andrewmvd/helmet-detection")
        
        # Download latest version
        path = kagglehub.dataset_download("andrewmvd/helmet-detection")
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"📁 Path to dataset files: {path}")
        
        return path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Note: You may need to:")
        print("1. Install Kaggle CLI: pip install kaggle")
        print("2. Set up Kaggle credentials: https://www.kaggle.com/docs/api")
        print("3. Or download manually from: https://www.kaggle.com/datasets/andrewmvd/helmet-detection")
        return None


def organize_dataset(dataset_path):
    """Organize the downloaded dataset into our project structure."""
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Dataset path not found")
        return False
    
    print(f"\n📁 Organizing dataset from: {dataset_path}")
    
    # Clear existing data (backup first)
    backup_dir = "data_backup"
    if os.path.exists("data/images") and os.listdir("data/images"):
        print("📦 Backing up existing data...")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree("data", backup_dir)
        print(f"✅ Backup created: {backup_dir}")
    
    # Clear data directories
    for dir_name in ["data/images", "data/annotations"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)
    
    # Find and copy images
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(dataset_path, "**", pattern), recursive=True))
    
    if image_files:
        print(f"📸 Found {len(image_files)} images")
        for img_file in image_files:
            filename = os.path.basename(img_file)
            dest_path = os.path.join("data/images", filename)
            shutil.copy2(img_file, dest_path)
        print(f"✅ Copied {len(image_files)} images to data/images/")
    else:
        print("⚠️ No images found in dataset")
    
    # Find and copy XML annotations
    xml_files = glob.glob(os.path.join(dataset_path, "**", "*.xml"), recursive=True)
    
    if xml_files:
        print(f"📋 Found {len(xml_files)} XML annotation files")
        for xml_file in xml_files:
            filename = os.path.basename(xml_file)
            dest_path = os.path.join("data/annotations", filename)
            shutil.copy2(xml_file, dest_path)
        print(f"✅ Copied {len(xml_files)} annotations to data/annotations/")
    else:
        print("⚠️ No XML annotations found in dataset")
    
    # Show dataset structure
    print(f"\n📊 Dataset Summary:")
    image_count = len([f for f in os.listdir("data/images") if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    xml_count = len([f for f in os.listdir("data/annotations") if f.endswith('.xml')])
    
    print(f"   - Images: {image_count}")
    print(f"   - XML Annotations: {xml_count}")
    
    return image_count > 0 and xml_count > 0


def convert_annotations():
    """Convert XML annotations to YOLO format."""
    print("\n🔄 Converting XML annotations to YOLO format...")
    
    try:
        # Import and run our XML to YOLO converter
        import sys
        sys.path.append("app")
        from xml_to_yolo_converter import XMLToYOLOConverter
        
        converter = XMLToYOLOConverter()
        converter.convert_dataset("data/annotations", "data/labels")
        
        # Check results
        label_count = len([f for f in os.listdir("data/labels") if f.endswith('.txt')])
        print(f"✅ Created {label_count} YOLO label files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting annotations: {e}")
        return False


def main():
    """Main function to download and organize the dataset."""
    
    # Download dataset
    dataset_path = download_helmet_dataset()
    
    if not dataset_path:
        print("\n❌ Dataset download failed")
        return
    
    # Organize dataset
    success = organize_dataset(dataset_path)
    
    if not success:
        print("\n❌ Dataset organization failed")
        return
    
    # Convert annotations
    convert_success = convert_annotations()
    
    if not convert_success:
        print("\n⚠️ Annotation conversion failed, but you can run it manually:")
        print("python3 app/xml_to_yolo_converter.py")
    
    # Final status
    print(f"\n🎉 Dataset setup complete!")
    
    # Check final counts
    image_count = len([f for f in os.listdir("data/images") if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    label_count = len([f for f in os.listdir("data/labels") if f.endswith('.txt')])
    
    print(f"\n📈 Final Dataset Status:")
    print(f"   - Training Images: {image_count}")
    print(f"   - YOLO Labels: {label_count}")
    
    if image_count > 100 and label_count > 100:
        print(f"\n🏋️ Ready for training!")
        print(f"Next steps:")
        print(f"1. Train model: python3 train_model.py")
        print(f"2. Test detection: python3 app/main.py --image data/images/sample.jpg")
    else:
        print(f"\n⚠️ Dataset seems small. Consider checking:")
        print(f"1. Dataset downloaded correctly")
        print(f"2. Annotations converted properly")
        print(f"3. File paths and formats")


if __name__ == "__main__":
    main()
