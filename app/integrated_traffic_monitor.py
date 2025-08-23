#!/usr/bin/env python3
"""
Integrated Traffic Monitoring System
Combines helmet detection and license plate recognition for traffic safety enforcement.
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

from app.helmet_detector import HelmetDetector
from app.license_plate_detector import LicensePlateDetector

class IntegratedTrafficMonitor:
    """
    Integrated system for helmet detection and license plate recognition.
    Automatically tracks license plates of motorbike riders without helmets.
    """
    
    def __init__(self, 
                 helmet_model_path=None,
                 license_plate_model_path=None,
                 device="auto",
                 helmet_confidence=0.5,
                 plate_confidence=0.5,
                 save_violations=True,
                 output_dir="violations"):
        """
        Initialize integrated traffic monitoring system.
        
        Args:
            helmet_model_path (str): Path to helmet detection model
            license_plate_model_path (str): Path to license plate detection model
            device (str): Device to use ('auto', 'cuda', 'cpu')
            helmet_confidence (float): Confidence threshold for helmet detection
            plate_confidence (float): Confidence threshold for license plate detection
            save_violations (bool): Whether to save violation images
            output_dir (str): Directory to save violation records
        """
        self.device = self._get_device(device)
        self.save_violations = save_violations
        self.output_dir = Path(output_dir)
        
        if self.save_violations:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir / "images", exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.helmet_detector = HelmetDetector(
            model_path=helmet_model_path,
            device=self.device,
            confidence=helmet_confidence
        )
        
        self.license_plate_detector = LicensePlateDetector(
            model_path=license_plate_model_path,
            device=self.device,
            confidence_threshold=plate_confidence
        )
        
        # Violation tracking
        self.violation_count = 0
        self.session_violations = []
        
        self.logger.info(f"Integrated Traffic Monitor initialized on {self.device}")
        
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
    
    def detect_violations(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect helmet violations and track license plates.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: (annotated_image, violations_list)
        """
        # Step 1: Detect helmets and persons
        # For now, we'll process the image directly since the helmet detector expects file path
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp_file.close()  # Close the file handle on Windows
        cv2.imwrite(tmp_file.name, image)
        helmet_detections = self.helmet_detector.detect(tmp_file.name)
        try:
            os.unlink(tmp_file.name)  # Clean up temp file
        except PermissionError:
            pass  # File may still be in use, will be cleaned up later
        
        violations = []
        annotated_image = image.copy()
        
        # Step 2: Process helmet detections to find violations
        riders_without_helmets = []
        
        for detection in helmet_detections:
            bbox_xyxy = detection['bbox_xyxy']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox_xyxy]
            
            if class_name == "Without Helmet":
                riders_without_helmets.append({
                    'bbox': bbox_xyxy,
                    'confidence': confidence,
                    'person_roi': image[y1:y2, x1:x2]
                })
                
                # Draw red box for violation
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated_image, f"VIOLATION: No Helmet ({confidence:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif class_name == "With Helmet":
                # Draw green box for compliance
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"OK: Helmet ({confidence:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Step 3: For each violation, try to detect license plates
        if riders_without_helmets:
            plate_image, plate_detections = self.license_plate_detector.detect_plates(image, draw_boxes=False)
            
            for rider in riders_without_helmets:
                violation = {
                    'timestamp': datetime.now().isoformat(),
                    'rider_bbox': [float(x) for x in rider['bbox']],
                    'rider_confidence': float(rider['confidence']),
                    'license_plates': []
                }
                
                # Find license plates in the image
                for plate_det in plate_detections:
                    plate_bbox = plate_det['bbox']
                    plate_conf = plate_det['confidence']
                    
                    # Try to extract plate text
                    if 'plate_image' in plate_det:
                        plate_text = self.license_plate_detector.extract_plate_text(plate_det['plate_image'])
                        
                        plate_info = {
                            'bbox': [float(x) for x in plate_bbox],
                            'confidence': float(plate_conf),
                            'text': plate_text
                        }
                        violation['license_plates'].append(plate_info)
                        
                        # Draw license plate detection
                        px1, py1, px2, py2 = plate_bbox
                        cv2.rectangle(annotated_image, (px1, py1), (px2, py2), (255, 0, 0), 2)
                        cv2.putText(annotated_image, f"Plate: {plate_text} ({plate_conf:.2f})", 
                                   (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                violations.append(violation)
        
        # Add summary information
        if violations:
            cv2.putText(annotated_image, f"VIOLATIONS DETECTED: {len(violations)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Save violations if enabled
            if self.save_violations:
                self._save_violation_record(annotated_image, violations)
        else:
            cv2.putText(annotated_image, "NO VIOLATIONS", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        return annotated_image, violations
    
    def _save_violation_record(self, image: np.ndarray, violations: List[Dict]):
        """Save violation record with image and metadata."""
        if not self.save_violations:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.violation_count += 1
        
        # Save image
        image_filename = f"violation_{self.violation_count:04d}_{timestamp}.jpg"
        image_path = self.output_dir / "images" / image_filename
        cv2.imwrite(str(image_path), image)
        
        # Save metadata
        violation_record = {
            'violation_id': self.violation_count,
            'timestamp': timestamp,
            'image_file': image_filename,
            'violations': violations,
            'total_violations': len(violations)
        }
        
        # Append to session violations
        self.session_violations.append(violation_record)
        
        # Save to JSON file
        json_path = self.output_dir / f"violation_{self.violation_count:04d}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(violation_record, f, indent=2)
        
        self.logger.info(f"Violation record saved: {json_path}")
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """
        Process video for traffic violations.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            display (bool): Whether to display video while processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Processing video: {width}x{height} @ {fps}fps ({total_frames} frames)")
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_violations = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame for violations
                processed_frame, violations = self.detect_violations(frame)
                total_violations += len(violations)
                
                # Add frame counter
                cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                           (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Save frame if output video specified
                if out:
                    out.write(processed_frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Traffic Monitor', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_count % (fps * 5) == 0:  # Every 5 seconds
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames. "
                                   f"Violations: {total_violations}")
        
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        self.logger.info(f"Video processing complete. Total violations: {total_violations}")
        
        # Save session summary
        self._save_session_summary(video_path, total_frames, total_violations)
    
    def process_image(self, image_path: str, output_path: str = None, display: bool = True):
        """
        Process single image for traffic violations.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image (optional)
            display (bool): Whether to display result
        """
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Error loading image: {image_path}")
            return
        
        # Process image
        processed_image, violations = self.detect_violations(image)
        
        # Save output if requested
        if output_path:
            cv2.imwrite(output_path, processed_image)
            self.logger.info(f"Output saved to: {output_path}")
        
        # Display if requested
        if display:
            cv2.imshow('Traffic Monitor', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return processed_image, violations
    
    def _save_session_summary(self, video_path: str, total_frames: int, total_violations: int):
        """Save session summary."""
        summary = {
            'session_timestamp': datetime.now().isoformat(),
            'video_path': video_path,
            'total_frames': total_frames,
            'total_violations': total_violations,
            'violations_per_frame': total_violations / total_frames if total_frames > 0 else 0,
            'violation_records': self.session_violations
        }
        
        summary_path = self.output_dir / f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Session summary saved: {summary_path}")

def main():
    """Main function for testing the integrated system."""
    print("🚀 Integrated Traffic Monitoring System")
    print("=" * 50)
    
    # Initialize monitor
    monitor = IntegratedTrafficMonitor(
        helmet_model_path="models/helmet_yolov8_gpu.pt",  # Your trained helmet model
        license_plate_model_path=None,  # Will use base model for now
        helmet_confidence=0.3,  # Lower confidence for better detection
        plate_confidence=0.3,
        save_violations=True
    )
    
    # Test with sample images
    test_images = [
        "data/images/BikesHelmets1.png",  # Should have helmet
        "data/images/BikesHelmets3.png",  # Test image
        "data/images/BikesHelmets4.png",  # Test image
        "data/images/h.jpg",
        "data/images/n1.jpg", 
        "data/images/n2.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n🔍 Processing: {img_path}")
            result, violations = monitor.process_image(img_path, display=False)
            print(f"   Violations found: {len(violations)}")
            
            for i, violation in enumerate(violations):
                print(f"   Violation {i+1}:")
                print(f"     License plates detected: {len(violation['license_plates'])}")
                for plate in violation['license_plates']:
                    print(f"       Plate text: {plate['text']} (conf: {plate['confidence']:.2f})")

if __name__ == "__main__":
    main()
