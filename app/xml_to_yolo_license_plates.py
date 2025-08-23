#!/usr/bin/env python3
"""
XML to YOLO Converter for License Plate Dataset
Converts Pascal VOC XML annotations to YOLO format for license plate detection.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import cv2
import glob
from tqdm import tqdm

class LicensePlateXMLConverter:
    """Convert XML annotations to YOLO format for license plate dataset."""
    
    def __init__(self, xml_dir, images_dir, output_dir=None):
        """
        Initialize converter.
        
        Args:
            xml_dir (str): Directory containing XML annotation files
            images_dir (str): Directory containing corresponding images
            output_dir (str): Output directory for YOLO labels (optional)
        """
        self.xml_dir = Path(xml_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("data/license_plates")
        
        # Create output directories
        self.labels_dir = self.output_dir / "labels"
        self.yolo_images_dir = self.output_dir / "images"
        
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.yolo_images_dir, exist_ok=True)
        
        self.class_names = ['license_plate']  # Single class for license plates
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"📁 XML directory: {self.xml_dir}")
        print(f"📁 Images directory: {self.images_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🏷️  Classes: {self.class_names}")
    
    def parse_xml_file(self, xml_path):
        """
        Parse XML annotation file and extract bounding box information.
        
        Args:
            xml_path (str): Path to XML file
            
        Returns:
            dict: Parsed annotation data
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image information
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            annotations = []
            
            # Parse all objects (license plates)
            for obj in root.findall('object'):
                class_name = obj.find('name').text.lower()
                
                # Map various license plate class names to our standard
                if any(keyword in class_name for keyword in ['plate', 'number', 'license']):
                    class_name = 'license_plate'
                
                if class_name in self.class_mapping:
                    bbox = obj.find('bndbox')
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    # Convert to YOLO format (normalized center coordinates + width/height)
                    x_center = (xmin + xmax) / 2.0 / width
                    y_center = (ymin + ymax) / 2.0 / height
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height
                    
                    # Ensure coordinates are within bounds
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    bbox_width = max(0, min(1, bbox_width))
                    bbox_height = max(0, min(1, bbox_height))
                    
                    annotations.append({
                        'class_id': self.class_mapping[class_name],
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': bbox_width,
                        'height': bbox_height
                    })
            
            return {
                'filename': filename,
                'width': width,
                'height': height,
                'annotations': annotations
            }
            
        except Exception as e:
            print(f"❌ Error parsing {xml_path}: {e}")
            return None
    
    def convert_dataset(self, copy_images=True):
        """
        Convert entire dataset from XML to YOLO format.
        
        Args:
            copy_images (bool): Whether to copy images to output directory
        """
        xml_files = list(self.xml_dir.glob("*.xml"))
        
        if not xml_files:
            print(f"❌ No XML files found in {self.xml_dir}")
            return
        
        print(f"🔄 Converting {len(xml_files)} XML files...")
        
        converted_count = 0
        skipped_count = 0
        
        for xml_file in tqdm(xml_files, desc="Converting annotations"):
            # Parse XML
            parsed_data = self.parse_xml_file(xml_file)
            
            if parsed_data is None:
                skipped_count += 1
                continue
            
            # Find corresponding image
            image_name = parsed_data['filename']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_path = None
            
            for ext in image_extensions:
                potential_path = self.images_dir / image_name.replace(Path(image_name).suffix, ext)
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path is None:
                print(f"⚠️  Image not found for {xml_file.name}")
                skipped_count += 1
                continue
            
            # Copy image if requested
            if copy_images:
                import shutil
                target_image_path = self.yolo_images_dir / image_path.name
                if not target_image_path.exists():
                    shutil.copy2(image_path, target_image_path)
            
            # Create YOLO annotation file
            label_filename = xml_file.stem + '.txt'
            label_path = self.labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for ann in parsed_data['annotations']:
                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                    f.write(line)
            
            converted_count += 1
        
        print(f"✅ Conversion complete!")
        print(f"   📊 Converted: {converted_count} files")
        print(f"   ⚠️  Skipped: {skipped_count} files")
        
        # Create dataset configuration
        self.create_dataset_config()
        
        # Generate statistics
        self.generate_statistics()
    
    def create_dataset_config(self):
        """Create YOLO dataset configuration file."""
        config_content = f"""# License Plate Detection Dataset Configuration
train: {self.yolo_images_dir.absolute()}
val: {self.yolo_images_dir.absolute()}
test: {self.yolo_images_dir.absolute()}

nc: {len(self.class_names)}
names:
"""
        for i, name in enumerate(self.class_names):
            config_content += f"  {i}: {name}\n"
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"📝 Dataset config saved to: {config_path}")
    
    def generate_statistics(self):
        """Generate dataset statistics."""
        label_files = list(self.labels_dir.glob("*.txt"))
        image_files = list(self.yolo_images_dir.glob("*"))
        
        total_annotations = 0
        class_counts = {name: 0 for name in self.class_names}
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                total_annotations += len(lines)
                
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if class_id < len(self.class_names):
                            class_counts[self.class_names[class_id]] += 1
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Images: {len(image_files)}")
        print(f"   Label files: {len(label_files)}")
        print(f"   Total annotations: {total_annotations}")
        print(f"   Class distribution:")
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count}")
    
    def validate_dataset(self):
        """Validate the converted dataset."""
        print("🔍 Validating dataset...")
        
        label_files = list(self.labels_dir.glob("*.txt"))
        image_files = list(self.yolo_images_dir.glob("*"))
        
        # Check for orphan files
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        orphan_images = image_stems - label_stems
        orphan_labels = label_stems - image_stems
        
        if orphan_images:
            print(f"⚠️  Images without labels: {len(orphan_images)}")
        
        if orphan_labels:
            print(f"⚠️  Labels without images: {len(orphan_labels)}")
        
        # Validate label format
        invalid_labels = []
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) != 5:
                                invalid_labels.append(f"{label_file.name}:{line_num}")
                            else:
                                # Check if values are valid
                                class_id = int(parts[0])
                                coords = [float(x) for x in parts[1:]]
                                
                                if class_id >= len(self.class_names):
                                    invalid_labels.append(f"{label_file.name}:{line_num} (invalid class)")
                                
                                if not all(0 <= coord <= 1 for coord in coords):
                                    invalid_labels.append(f"{label_file.name}:{line_num} (coords out of bounds)")
            
            except Exception as e:
                invalid_labels.append(f"{label_file.name} (error: {e})")
        
        if invalid_labels:
            print(f"❌ Invalid label entries found: {len(invalid_labels)}")
            for invalid in invalid_labels[:10]:  # Show first 10
                print(f"   {invalid}")
            if len(invalid_labels) > 10:
                print(f"   ... and {len(invalid_labels) - 10} more")
        else:
            print("✅ All labels are valid!")

def main():
    """Main function to convert license plate dataset."""
    # Default paths - adjust as needed
    xml_dir = "data/license_plates/annotations"  # Where XML files are stored
    images_dir = "data/license_plates/source_images"  # Where source images are stored
    output_dir = "data/license_plates"  # Where to save YOLO format data
    
    print("🚀 License Plate Dataset Converter")
    print("=" * 50)
    
    # Check if directories exist
    if not os.path.exists(xml_dir):
        print(f"❌ XML directory not found: {xml_dir}")
        print("Please download the license plate dataset and extract it to the correct location.")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        print("Please ensure images are in the correct location.")
        return
    
    # Create converter and process dataset
    converter = LicensePlateXMLConverter(xml_dir, images_dir, output_dir)
    converter.convert_dataset(copy_images=True)
    converter.validate_dataset()
    
    print("\n✅ License plate dataset conversion complete!")
    print(f"📁 YOLO format data saved to: {output_dir}")
    print("🎯 Ready for training!")

if __name__ == "__main__":
    main()
