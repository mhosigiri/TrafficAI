"""
XML to YOLO Format Converter for Helmet Detection
Converts Pascal VOC XML annotations to YOLO format for training YOLOv8 models.
"""

import os
import shutil
import yaml
import glob
from xml.dom import minidom
from pathlib import Path


class XMLToYOLOConverter:
    """Converts XML annotations to YOLO format for helmet detection."""
    
    def __init__(self, class_mapping=None):
        """Initialize converter with class mapping."""
        self.class_mapping = class_mapping or {"Without Helmet": 0, "With Helmet": 1}
    
    def get_text(self, tag, parent):
        """Helper function to extract text from an XML tag."""
        return parent.getElementsByTagName(tag)[0].firstChild.data
    
    def convert_coordinates(self, size, box):
        """Convert Pascal VOC bounding box format to YOLO format.
        
        Args:
            size: (width, height) of the image
            box: (xmin, xmax, ymin, ymax) bounding box coordinates
            
        Returns:
            (x_center, y_center, width, height) normalized to [0,1]
        """
        dw, dh = 1.0 / size[0], 1.0 / size[1]
        x = (box[0] + box[1]) / 2.0 * dw
        y = (box[2] + box[3]) / 2.0 * dh
        w = (box[1] - box[0]) * dw
        h = (box[3] - box[2]) * dh
        return x, y, w, h
    
    def convert_xml_file(self, xml_path, output_path):
        """Convert a single XML file to YOLO format.
        
        Args:
            xml_path: Path to input XML file
            output_path: Path to output TXT file
        """
        try:
            xmldoc = minidom.parse(xml_path)
            size = xmldoc.getElementsByTagName("size")[0]
            
            width = int(self.get_text("width", size))
            height = int(self.get_text("height", size))
            
            with open(output_path, "w") as f:
                for item in xmldoc.getElementsByTagName("object"):
                    class_name = self.get_text("name", item)
                    
                    if class_name not in self.class_mapping:
                        print(f"⚠️ Warning: Unknown label '{class_name}' in {xml_path}. Skipping.")
                        continue
                    
                    # Extract bounding box
                    bbox = item.getElementsByTagName("bndbox")[0]
                    xmin = float(self.get_text("xmin", bbox))
                    xmax = float(self.get_text("xmax", bbox))
                    ymin = float(self.get_text("ymin", bbox))
                    ymax = float(self.get_text("ymax", bbox))
                    
                    # Convert to YOLO format
                    bb = self.convert_coordinates((width, height), (xmin, xmax, ymin, ymax))
                    class_id = self.class_mapping[class_name]
                    
                    # Write to file
                    f.write(f"{class_id} " + " ".join(f"{x:.6f}" for x in bb) + "\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error converting {xml_path}: {e}")
            return False
    
    def convert_dataset(self, xml_folder, output_folder):
        """Convert all XML files in a folder to YOLO format.
        
        Args:
            xml_folder: Folder containing XML annotation files
            output_folder: Folder to save YOLO format labels
        """
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        xml_files = glob.glob(os.path.join(xml_folder, "*.xml"))
        
        if not xml_files:
            print(f"⚠️ No XML files found in {xml_folder}")
            return
        
        converted_count = 0
        for xml_path in xml_files:
            filename = os.path.basename(xml_path).replace(".xml", ".txt")
            output_path = os.path.join(output_folder, filename)
            
            if self.convert_xml_file(xml_path, output_path):
                converted_count += 1
                print(f"✅ Converted: {filename}")
        
        print(f"\n🎉 Successfully converted {converted_count}/{len(xml_files)} files")
    
    def create_dataset_yaml(self, dataset_path, train_path, val_path, test_path=None):
        """Create dataset.yaml file for YOLOv8 training.
        
        Args:
            dataset_path: Path to save dataset.yaml
            train_path: Path to training images
            val_path: Path to validation images
            test_path: Path to test images (optional)
        """
        data = {
            "train": str(train_path),
            "val": str(val_path),
            "nc": len(self.class_mapping),
            "names": list(self.class_mapping.keys())
        }
        
        if test_path:
            data["test"] = str(test_path)
        
        with open(dataset_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"✅ Created dataset.yaml at {dataset_path}")


def main():
    """Example usage of the converter."""
    converter = XMLToYOLOConverter()
    
    # Example paths (adjust these for your dataset)
    xml_folder = "data/annotations"
    labels_folder = "data/labels"
    images_folder = "data/images"
    
    if os.path.exists(xml_folder):
        print("🔄 Converting XML annotations to YOLO format...")
        converter.convert_dataset(xml_folder, labels_folder)
        
        # Create dataset.yaml
        converter.create_dataset_yaml(
            "config/dataset.yaml",
            images_folder,
            images_folder  # Using same folder for train/val in this example
        )
    else:
        print(f"❌ XML folder not found: {xml_folder}")
        print("Please place your XML annotation files in the data/annotations folder")


if __name__ == "__main__":
    main()
