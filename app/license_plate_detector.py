#!/usr/bin/env python3
"""
License Plate Detection Module
Based on YOLO object detection for automatic license plate recognition.
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
import logging
import platform

class LicensePlateDetector:
    """License plate detector using YOLO."""
    
    def __init__(self, model_path=None, device="auto", confidence_threshold=0.5):
        """
        Initialize license plate detector.
        
        Args:
            model_path (str): Path to trained license plate model
            device (str): Device to use ('auto', 'cuda', 'cpu')
            confidence_threshold (float): Minimum confidence for detections
        """
        self.device = self._get_device(device)
        self.confidence_threshold = confidence_threshold
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self._load_model(model_path)
        
        self.logger.info(f"License plate detector initialized on {self.device}")
        
    def _get_device(self, device):
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self, model_path):
        """Load the license plate detection model."""
        if model_path is None:
            # Use a pre-trained YOLO model or create a placeholder
            model_path = "yolov8n.pt"  # Will be replaced with custom license plate model
            self.logger.warning("No custom license plate model provided, using base YOLOv8")
        
        try:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                self.logger.info(f"Loaded custom model from {model_path}")
            else:
                model = YOLO("yolov8n.pt")  # Fallback to base model
                self.logger.warning(f"Model file {model_path} not found, using base YOLOv8")
            
            # Move model to device
            model.to(self.device)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def detect_plates(self, image, draw_boxes=True):
        """
        Detect license plates in an image.
        
        Args:
            image (np.ndarray): Input image
            draw_boxes (bool): Whether to draw detection boxes
            
        Returns:
            tuple: (annotated_image, detections)
        """
        try:
            # Run inference
            results = self.model(image, device=self.device, verbose=False)
            
            detections = []
            annotated_image = image.copy()
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Extract license plate region
                            plate_roi = image[y1:y2, x1:x2]
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class': int(cls),
                                'plate_image': plate_roi
                            }
                            detections.append(detection)
                            
                            if draw_boxes:
                                # Draw bounding box
                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw confidence label
                                label = f"License Plate: {conf:.2f}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 5), 
                                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            return annotated_image, detections
            
        except Exception as e:
            self.logger.error(f"Error during license plate detection: {e}")
            return image, []
    
    def extract_plate_text(self, plate_image):
        """
        Extract text from license plate image using OCR.
        
        Args:
            plate_image (np.ndarray): License plate image
            
        Returns:
            str: Extracted text
        """
        try:
            import pytesseract
            
            # Set Tesseract path for Windows
            if platform.system() == 'Windows':
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            # Apply image enhancement
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Configure tesseract for license plates
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Extract text
            text = pytesseract.image_to_string(gray, config=config).strip()
            return text
            
        except ImportError:
            self.logger.warning("pytesseract not available. Install with: pip install pytesseract")
            return "OCR_NOT_AVAILABLE"
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return "OCR_ERROR"
    
    def process_detections_with_ocr(self, detections):
        """
        Process detections and extract text from license plates.
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            list: Detections with added 'text' field
        """
        for detection in detections:
            if 'plate_image' in detection:
                detection['text'] = self.extract_plate_text(detection['plate_image'])
        
        return detections

def create_license_plate_dataset_config():
    """Create dataset configuration for license plate training."""
    config = {
        'train': 'data/license_plates/images',
        'val': 'data/license_plates/images',
        'test': 'data/license_plates/images',
        'nc': 1,  # number of classes
        'names': ['license_plate']
    }
    
    os.makedirs('config', exist_ok=True)
    
    with open('config/license_plate_dataset.yaml', 'w') as f:
        for key, value in config.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for i, name in enumerate(value):
                    f.write(f"  {i}: {name}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    return 'config/license_plate_dataset.yaml'

if __name__ == "__main__":
    # Test the license plate detector
    detector = LicensePlateDetector()
    
    # Create sample test
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "License Plate Detector Test", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    annotated, detections = detector.detect_plates(test_image)
    print(f"Detector initialized successfully. Found {len(detections)} detections.")
