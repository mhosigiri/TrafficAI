#!/usr/bin/env python3
"""
TrafficAI - Project Setup Script
Downloads sample data and sets up the project for helmet detection.
"""

import os
import urllib.request
import shutil
from pathlib import Path


def download_sample_image(url, filename):
    """Download a sample image for testing."""
    try:
        print(f"📥 Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False


def create_sample_mask():
    """Create a sample wrong lane mask."""
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple binary mask (640x640)
        mask = np.zeros((640, 640), dtype=np.uint8)
        
        # Mark left side as wrong lane (white = forbidden)
        mask[:, :320] = 255
        
        # Save mask
        mask_path = "data/masks/wrong_lane_mask.png"
        Image.fromarray(mask).save(mask_path)
        print(f"✅ Created sample mask: {mask_path}")
        return True
        
    except ImportError:
        print("⚠️ PIL not available, skipping mask creation")
        return False
    except Exception as e:
        print(f"❌ Failed to create mask: {e}")
        return False


def setup_directories():
    """Create necessary project directories."""
    directories = [
        "data/images",
        "data/labels", 
        "data/samples",
        "data/masks",
        "data/annotations",
        "models",
        "runs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")


def create_sample_annotation():
    """Create a sample XML annotation file."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <folder>images</folder>
    <filename>sample1.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>With Helmet</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
    <object>
        <name>Without Helmet</name>
        <bndbox>
            <xmin>300</xmin>
            <ymin>150</ymin>
            <xmax>400</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>"""
    
    with open("data/annotations/sample1.xml", "w") as f:
        f.write(sample_xml)
    print("✅ Created sample XML annotation")


def main():
    """Set up the TrafficAI project."""
    print("🚀 Setting up TrafficAI Project")
    print("=" * 40)
    
    # Create directories
    print("\n📁 Creating project structure...")
    setup_directories()
    
    # Create sample files
    print("\n📄 Creating sample files...")
    create_sample_annotation()
    create_sample_mask()
    
    # Sample images (you can add URLs to actual sample images here)
    sample_images = [
        # Add URLs to sample motorcycle/helmet images if available
        # ("https://example.com/sample1.jpg", "data/samples/sample1.jpg"),
    ]
    
    print("\n📸 Sample images:")
    if sample_images:
        for url, filename in sample_images:
            download_sample_image(url, filename)
    else:
        print("⚠️ No sample image URLs provided")
        print("Please add your own images to data/samples/ for testing")
    
    # Create a simple test image placeholder
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple placeholder image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(img).save("data/samples/placeholder.jpg")
        print("✅ Created placeholder image for testing")
        
    except ImportError:
        print("⚠️ PIL not available, skipping placeholder image")
    
    print(f"\n✅ Project setup complete!")
    print(f"\n📋 Next steps:")
    print(f"1. Install dependencies: pip install -r requirements.txt")
    print(f"2. Add your images to data/images/")
    print(f"3. Add XML annotations to data/annotations/")
    print(f"4. Convert annotations: python app/xml_to_yolo_converter.py")
    print(f"5. Train model: python train_model.py")
    print(f"6. Test detection: python app/main.py --image data/samples/your_image.jpg")


if __name__ == "__main__":
    main()
