#!/usr/bin/env python3
"""
TrafficAI Production Deployment Script
Complete traffic monitoring system with helmet detection and license plate tracking.
"""

import os
import sys
import cv2
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import our custom modules
from app.integrated_traffic_monitor import IntegratedTrafficMonitor
from app.helmet_detector import HelmetDetector
from app.license_plate_detector import LicensePlateDetector

class TrafficMonitorDeployment:
    """Production-ready traffic monitoring deployment."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize deployment system.
        
        Args:
            config_path: Path to deployment configuration JSON
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize the monitoring system
        self.monitor = self._initialize_monitor()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'total_violations': 0,
            'helmets_detected': 0,
            'no_helmets_detected': 0,
            'plates_detected': 0,
            'session_start': datetime.now().isoformat()
        }
        
        self.logger.info("TrafficAI Deployment System Initialized")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load deployment configuration."""
        default_config = {
            'helmet_model': 'models/helmet_yolov8_gpu.pt',
            'license_plate_model': 'models/license_plate_yolov8.pt',
            'helmet_confidence': 0.3,
            'plate_confidence': 0.4,
            'device': 'auto',
            'save_violations': True,
            'output_dir': 'violations',
            'log_level': 'INFO',
            'enable_ocr': True,
            'video_output_fps': 30,
            'max_detections': 100
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, self.config['log_level'].upper())
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        log_file = f"logs/traffic_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def _initialize_monitor(self) -> IntegratedTrafficMonitor:
        """Initialize the integrated monitoring system."""
        try:
            # Check if models exist
            if not os.path.exists(self.config['helmet_model']):
                self.logger.error(f"Helmet model not found: {self.config['helmet_model']}")
                raise FileNotFoundError(f"Helmet model not found: {self.config['helmet_model']}")
            
            # License plate model is optional
            license_plate_model = None
            if os.path.exists(self.config['license_plate_model']):
                license_plate_model = self.config['license_plate_model']
                self.logger.info(f"Using license plate model: {license_plate_model}")
            else:
                self.logger.warning("License plate model not found, using base YOLO")
            
            monitor = IntegratedTrafficMonitor(
                helmet_model_path=self.config['helmet_model'],
                license_plate_model_path=license_plate_model,
                device=self.config['device'],
                helmet_confidence=self.config['helmet_confidence'],
                plate_confidence=self.config['plate_confidence'],
                save_violations=self.config['save_violations'],
                output_dir=self.config['output_dir']
            )
            
            return monitor
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitor: {e}")
            raise
    
    def process_live_camera(self, camera_id: int = 0, display: bool = True):
        """
        Process live camera feed.
        
        Args:
            camera_id: Camera device ID
            display: Whether to display live feed
        """
        self.logger.info(f"Starting live camera processing (camera {camera_id})")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {fps}fps")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame, violations = self.monitor.detect_violations(frame)
                
                # Update statistics
                self.stats['total_processed'] += 1
                self.stats['total_violations'] += len(violations)
                
                # Add statistics overlay
                self._add_stats_overlay(processed_frame)
                
                # Display if requested
                if display:
                    cv2.imshow('TrafficAI Monitor - Press Q to quit', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Log violations
                if violations:
                    self.logger.info(f"Frame {self.stats['total_processed']}: {len(violations)} violation(s) detected")
        
        except KeyboardInterrupt:
            self.logger.info("Camera processing interrupted by user")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            self._save_session_report()
    
    def process_video_file(self, video_path: str, output_path: str = None, display: bool = True):
        """
        Process a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
            display: Whether to display during processing
        """
        self.logger.info(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return
        
        # Use monitor's video processing with stats tracking
        self.monitor.process_video(video_path, output_path, display)
        
        # Save session report
        self._save_session_report()
    
    def process_image_batch(self, image_dir: str, output_dir: str = None):
        """
        Process a batch of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save processed images
        """
        self.logger.info(f"Processing image batch from: {image_dir}")
        
        if not os.path.exists(image_dir):
            self.logger.error(f"Image directory not found: {image_dir}")
            return
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in image_files:
            try:
                self.logger.info(f"Processing: {img_path.name}")
                
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    self.logger.warning(f"Failed to read image: {img_path}")
                    continue
                
                # Process
                processed_image, violations = self.monitor.detect_violations(image)
                
                # Update stats
                self.stats['total_processed'] += 1
                self.stats['total_violations'] += len(violations)
                
                # Save if output directory specified
                if output_dir:
                    output_path = Path(output_dir) / f"processed_{img_path.name}"
                    cv2.imwrite(str(output_path), processed_image)
                    
                    # Save violation details
                    if violations:
                        json_path = output_path.with_suffix('.json')
                        with open(json_path, 'w') as f:
                            json.dump({
                                'image': img_path.name,
                                'violations': violations,
                                'timestamp': datetime.now().isoformat()
                            }, f, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
        
        self._save_session_report()
    
    def _add_stats_overlay(self, frame: np.ndarray):
        """Add statistics overlay to frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        stats_text = [
            f"Frames: {self.stats['total_processed']}",
            f"Violations: {self.stats['total_violations']}",
            f"Violation Rate: {self.stats['total_violations']/max(1, self.stats['total_processed'])*100:.1f}%",
            f"Session: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        y_offset = 35
        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def _save_session_report(self):
        """Save session statistics report."""
        self.stats['session_end'] = datetime.now().isoformat()
        
        report_path = f"logs/session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        self.logger.info(f"Session report saved: {report_path}")
        self.logger.info(f"Total violations detected: {self.stats['total_violations']}")
    
    def run_health_check(self) -> bool:
        """Run system health check."""
        self.logger.info("Running system health check...")
        
        checks = {
            'helmet_model': os.path.exists(self.config['helmet_model']),
            'gpu_available': torch.cuda.is_available(),
            'output_directory': os.access(self.config['output_dir'], os.W_OK),
            'helmet_detector': self.monitor.helmet_detector is not None,
            'license_plate_detector': self.monitor.license_plate_detector is not None
        }
        
        # Test models with dummy data
        try:
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            _, _ = self.monitor.detect_violations(test_image)
            checks['model_inference'] = True
        except:
            checks['model_inference'] = False
        
        # Print results
        all_passed = all(checks.values())
        
        self.logger.info("Health Check Results:")
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            self.logger.info(f"  {check}: {status}")
        
        return all_passed

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="TrafficAI Deployment System")
    parser.add_argument('--mode', choices=['camera', 'video', 'batch', 'health'], 
                       default='health', help='Operation mode')
    parser.add_argument('--input', help='Input video/image path or directory')
    parser.add_argument('--output', help='Output path or directory')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--config', help='Configuration JSON file')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    print("🚀 TrafficAI Production Deployment")
    print("=" * 50)
    
    # Initialize deployment system
    try:
        deployment = TrafficMonitorDeployment(args.config)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return 1
    
    # Execute based on mode
    if args.mode == 'health':
        success = deployment.run_health_check()
        return 0 if success else 1
    
    elif args.mode == 'camera':
        deployment.process_live_camera(
            camera_id=args.camera, 
            display=not args.no_display
        )
    
    elif args.mode == 'video':
        if not args.input:
            print("❌ Video mode requires --input parameter")
            return 1
        deployment.process_video_file(
            args.input, 
            args.output, 
            display=not args.no_display
        )
    
    elif args.mode == 'batch':
        if not args.input:
            print("❌ Batch mode requires --input parameter")
            return 1
        deployment.process_image_batch(args.input, args.output)
    
    print("\n✅ Deployment completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
