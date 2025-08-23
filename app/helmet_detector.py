"""
Helmet Detection using YOLOv8
Main inference module for detecting helmets in images.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os


class HelmetDetector:
    """YOLOv8-based helmet detector."""
    
    def __init__(self, model_path="models/helmet_yolov8.pt", confidence=0.5):
        """Initialize the helmet detector.
        
        Args:
            model_path: Path to the trained YOLOv8 model
            confidence: Minimum confidence threshold for detections
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.class_names = ["Without Helmet", "With Helmet"]
        
        self.load_model()
    
    def load_model(self):
        """Load the YOLOv8 model."""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Loaded custom model from {self.model_path}")
            else:
                # Use pretrained YOLOv8 nano model as fallback
                print(f"⚠️ Custom model not found at {self.model_path}")
                print("Using YOLOv8 nano model as fallback (will detect persons only)")
                self.model = YOLO('yolov8n.pt')
                self.class_names = ["person"]  # Standard COCO class
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect(self, image_path):
        """Detect helmets in an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detection dictionaries with bbox, class, and confidence
        """
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        try:
            # Run inference
            results = self.model(image_path, conf=self.confidence)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Convert to center coordinates and width/height
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        detection = {
                            'bbox': [x_center, y_center, width, height],
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'class_id': cls,
                            'class_name': self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}",
                            'confidence': conf,
                            'is_helmet': cls == 1 if len(self.class_names) == 2 else False  # With Helmet = class 1
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return []
    
    def annotate_image(self, image_path, detections, output_path=None):
        """Annotate image with detection results.
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            output_path: Path to save annotated image (optional)
            
        Returns:
            Annotated image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return None
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox_xyxy']]
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Choose color based on helmet presence
            if det.get('is_helmet', False):
                color = (0, 255, 0)  # Green for helmet
                label = f"With Helmet: {confidence:.2f}"
            elif class_name == "Without Helmet":
                color = (0, 0, 255)  # Red for no helmet
                label = f"No Helmet: {confidence:.2f}"
            else:
                color = (255, 0, 0)  # Blue for other detections
                label = f"{class_name}: {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save annotated image
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"💾 Saved annotated image: {output_path}")
        
        return image
    
    def process_image(self, image_path, save_result=True):
        """Complete processing pipeline for a single image.
        
        Args:
            image_path: Path to input image
            save_result: Whether to save the annotated result
            
        Returns:
            Dictionary with detections and summary
        """
        print(f"\n🔍 Processing: {image_path}")
        
        # Run detection
        detections = self.detect(image_path)
        
        # Generate summary
        total_detections = len(detections)
        helmet_count = sum(1 for det in detections if det.get('is_helmet', False))
        no_helmet_count = sum(1 for det in detections if det['class_name'] == "Without Helmet")
        
        summary = {
            'image_path': image_path,
            'total_detections': total_detections,
            'helmet_count': helmet_count,
            'no_helmet_count': no_helmet_count,
            'detections': detections
        }
        
        # Print results
        print(f"📊 Results:")
        print(f"   - Total detections: {total_detections}")
        print(f"   - With helmet: {helmet_count}")
        print(f"   - Without helmet: {no_helmet_count}")
        
        # Save annotated image
        if save_result and detections:
            output_path = str(Path(image_path).with_suffix('')) + "_annotated.jpg"
            self.annotate_image(image_path, detections, output_path)
        
        return summary


def main():
    """Example usage of the helmet detector."""
    # Initialize detector
    detector = HelmetDetector()
    
    # Example processing
    sample_images = ["data/samples/sample1.jpg", "data/samples/sample2.jpg"]
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            result = detector.process_image(image_path)
        else:
            print(f"Sample image not found: {image_path}")
    
    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()
