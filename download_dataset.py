#!/usr/bin/env python3
"""
Download and prepare helmet detection training data
This script downloads a public helmet detection dataset and prepares it for training.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json

def download_file(url, filename, description="Downloading"):
    """Download a file with progress indication."""
    try:
        print(f"📥 {description}: {filename}")
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size / total_size) * 100)
                print(f"\r   Progress: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def create_sample_images():
    """Create sample images using OpenCV for testing if no dataset is available."""
    try:
        import cv2
        import numpy as np
        
        print("🎨 Creating sample training images...")
        
        # Create sample images with different scenarios
        samples = [
            ("rider_with_helmet_1.jpg", "Rider with helmet"),
            ("rider_with_helmet_2.jpg", "Rider with helmet - side view"), 
            ("rider_without_helmet_1.jpg", "Rider without helmet"),
            ("rider_without_helmet_2.jpg", "Rider without helmet - back view"),
            ("multiple_riders.jpg", "Multiple riders mixed")
        ]
        
        for i, (filename, description) in enumerate(samples):
            # Create a synthetic image (640x480)
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Add some basic shapes to simulate a motorcycle scene
            # Background
            cv2.rectangle(img, (0, 300), (640, 480), (100, 150, 100), -1)  # Road
            cv2.rectangle(img, (0, 0), (640, 300), (150, 200, 255), -1)    # Sky
            
            # Motorcycle (simple rectangle)
            cv2.rectangle(img, (250, 350), (400, 420), (50, 50, 50), -1)
            cv2.circle(img, (270, 420), 20, (30, 30, 30), -1)  # Wheel
            cv2.circle(img, (380, 420), 20, (30, 30, 30), -1)  # Wheel
            
            # Rider (simple shapes)
            cv2.rectangle(img, (300, 280), (350, 350), (100, 100, 200), -1)  # Body
            cv2.circle(img, (325, 260), 25, (150, 120, 100), -1)  # Head
            
            # Helmet (for with-helmet images)
            if "with_helmet" in filename:
                cv2.circle(img, (325, 260), 30, (200, 200, 200), 3)  # Helmet outline
                cv2.circle(img, (325, 260), 28, (50, 50, 50), -1)    # Helmet
            
            # Save image
            image_path = f"data/images/{filename}"
            cv2.imwrite(image_path, img)
            print(f"✅ Created sample: {filename} - {description}")
            
            # Create corresponding label file
            label_path = f"data/labels/{filename.replace('.jpg', '.txt')}"
            with open(label_path, 'w') as f:
                if "with_helmet" in filename:
                    # Head with helmet (class 1)
                    f.write("1 0.508 0.542 0.078 0.104\n")  # Normalized bbox coordinates
                elif "without_helmet" in filename:
                    # Head without helmet (class 0)
                    f.write("0 0.508 0.542 0.078 0.104\n")
                elif "multiple" in filename:
                    # Multiple riders - mixed scenario
                    f.write("1 0.400 0.542 0.078 0.104\n")  # With helmet
                    f.write("0 0.600 0.542 0.078 0.104\n")  # Without helmet
            
            print(f"✅ Created label: {filename.replace('.jpg', '.txt')}")
        
        return True
        
    except ImportError:
        print("⚠️ OpenCV not available for creating sample images")
        return False
    except Exception as e:
        print(f"❌ Error creating sample images: {e}")
        return False

def setup_roboflow_dataset():
    """Set up instructions for downloading from Roboflow (popular dataset source)."""
    print("\n🤖 Setting up Roboflow Dataset Integration")
    print("=" * 50)
    
    roboflow_script = '''
# Install roboflow if you want to use their datasets
# pip install roboflow

from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="YOUR_API_KEY")

# Download a helmet detection dataset
project = rf.workspace("your-workspace").project("helmet-detection")
dataset = project.version(1).download("yolov8")
'''
    
    with open("data/roboflow_example.py", "w") as f:
        f.write(roboflow_script)
    
    print("✅ Created roboflow_example.py")
    print("📋 To use Roboflow datasets:")
    print("1. Sign up at https://roboflow.com")
    print("2. Find a helmet detection dataset")
    print("3. Get your API key")
    print("4. Modify and run data/roboflow_example.py")

def download_public_dataset():
    """Try to download a public helmet detection dataset."""
    print("\n📦 Looking for public helmet detection datasets...")
    
    # Note: Most good datasets require registration/API keys
    # We'll provide instructions for manual download
    
    dataset_sources = [
        {
            "name": "Kaggle Helmet Detection Dataset",
            "url": "https://www.kaggle.com/datasets/andrewmvd/helmet-detection",
            "instructions": [
                "1. Go to: https://www.kaggle.com/datasets/andrewmvd/helmet-detection",
                "2. Click 'Download' (requires Kaggle account)",
                "3. Extract the zip file",
                "4. Copy images to data/images/",
                "5. Copy annotations to data/annotations/",
                "6. Run: python3 app/xml_to_yolo_converter.py"
            ]
        },
        {
            "name": "Roboflow Helmet Detection",
            "url": "https://universe.roboflow.com/search?q=helmet%20detection",
            "instructions": [
                "1. Browse datasets at: https://universe.roboflow.com",
                "2. Search for 'helmet detection'",
                "3. Choose a dataset and sign up",
                "4. Download in YOLOv8 format",
                "5. Extract to data/ folder"
            ]
        },
        {
            "name": "Custom Dataset Collection",
            "url": "manual",
            "instructions": [
                "1. Collect motorcycle/scooter images from traffic cameras",
                "2. Use labelImg tool to create annotations",
                "3. Save in Pascal VOC format (XML)",
                "4. Place images in data/images/",
                "5. Place XML files in data/annotations/",
                "6. Convert: python3 app/xml_to_yolo_converter.py"
            ]
        }
    ]
    
    instructions_file = "data/DATASET_SOURCES.md"
    with open(instructions_file, "w") as f:
        f.write("# Helmet Detection Dataset Sources\n\n")
        
        for source in dataset_sources:
            f.write(f"## {source['name']}\n")
            f.write(f"**URL**: {source['url']}\n\n")
            f.write("**Instructions**:\n")
            for instruction in source['instructions']:
                f.write(f"  {instruction}\n")
            f.write("\n")
    
    print(f"✅ Created dataset guide: {instructions_file}")
    return True

def main():
    """Main function to set up training data."""
    print("🚀 TrafficAI - Training Data Setup")
    print("=" * 40)
    
    # Ensure directories exist
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/labels", exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)
    
    # Create sample training data
    print("\n📊 Creating sample training data...")
    sample_created = create_sample_images()
    
    # Set up dataset download instructions
    download_public_dataset()
    setup_roboflow_dataset()
    
    # Check current data status
    print(f"\n📈 Current Dataset Status:")
    image_count = len([f for f in os.listdir("data/images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_count = len([f for f in os.listdir("data/labels") if f.endswith('.txt')])
    xml_count = len([f for f in os.listdir("data/annotations") if f.endswith('.xml')])
    
    print(f"   - Images: {image_count}")
    print(f"   - YOLO Labels: {label_count}")
    print(f"   - XML Annotations: {xml_count}")
    
    if image_count > 0 and label_count > 0:
        print("✅ Ready for training!")
        print(f"\n🏋️ Next steps:")
        print(f"1. Train model: python3 train_model.py")
        print(f"2. Test detection: python3 app/main.py --image data/images/rider_with_helmet_1.jpg")
    else:
        print("⚠️ Need more training data")
        print(f"\n📋 To add real training data:")
        print(f"1. Check data/DATASET_SOURCES.md for dataset options")
        print(f"2. Download a helmet detection dataset")
        print(f"3. Place images in data/images/")
        print(f"4. Place annotations in data/annotations/ (XML) or data/labels/ (YOLO)")
        print(f"5. If using XML: python3 app/xml_to_yolo_converter.py")
        print(f"6. Train: python3 train_model.py")

if __name__ == "__main__":
    main()
