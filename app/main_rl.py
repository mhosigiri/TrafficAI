#!/usr/bin/env python3
"""
TrafficAI - Enhanced Main Application with Reinforcement Learning
Helmet detection with user feedback and continuous learning capabilities.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time

from .helmet_detector import HelmetDetector
from .rl_trainer import RLHelmetTrainer
from .feedback_interface import FeedbackInterface


class TrafficAIRL:
    """Enhanced TrafficAI with reinforcement learning capabilities."""
    
    def __init__(self, model_path: str = "models/helmet_yolov8.pt"):
        """Initialize TrafficAI with RL support.
        
        Args:
            model_path: Path to the helmet detection model
        """
        self.model_path = model_path
        self.detector = HelmetDetector(model_path)
        self.rl_trainer = RLHelmetTrainer(model_path)
        
        print("🚀 TrafficAI with Reinforcement Learning initialized")
    
    def detect_with_feedback_option(self, image_path: str, save_result: bool = True,
                                  interactive: bool = False) -> dict:
        """Detect helmets with option for user feedback.
        
        Args:
            image_path: Path to input image
            save_result: Whether to save annotated result
            interactive: Whether to prompt for user feedback
            
        Returns:
            Detection results with feedback option
        """
        print(f"\n🔍 Processing: {image_path}")
        
        # Run detection
        result = self.detector.process_image(image_path, save_result)
        
        # Show results
        print(f"📊 Detection Results:")
        print(f"   - Total detections: {result['total_detections']}")
        print(f"   - With helmet: {result['helmet_count']}")
        print(f"   - Without helmet: {result['no_helmet_count']}")
        
        # Interactive feedback option
        if interactive and result['detections']:
            print(f"\n❓ Would you like to provide feedback on these results?")
            print(f"   This helps improve the model through reinforcement learning.")
            
            feedback_choice = input("Provide feedback? (y/n): ").lower().strip()
            
            if feedback_choice == 'y':
                self.collect_interactive_feedback(image_path, result['detections'])
        
        return result
    
    def collect_interactive_feedback(self, image_path: str, predictions: list):
        """Collect user feedback interactively via command line.
        
        Args:
            image_path: Path to the image
            predictions: Model predictions
        """
        print(f"\n📝 Collecting feedback for {len(predictions)} detections:")
        
        corrections = []
        
        for i, pred in enumerate(predictions):
            print(f"\n🎯 Detection #{i+1}:")
            print(f"   Class: {pred['class_name']}")
            print(f"   Confidence: {pred['confidence']:.2f}")
            print(f"   Bbox: {[int(x) for x in pred['bbox_xyxy']]}")
            
            print(f"\n   Options:")
            print(f"   ✅ c - Correct (no changes needed)")
            print(f"   ❌ w - Wrong class (with helmet ↔ without helmet)")
            print(f"   🗑️  d - Delete (false positive)")
            print(f"   ⏭️  s - Skip this detection")
            
            choice = input(f"   Your choice: ").lower().strip()
            
            if choice == 'c':
                # Correct prediction - no correction needed
                continue
            elif choice == 'w':
                # Wrong class - flip the class
                corrected = pred.copy()
                if pred['class_name'] == "With Helmet":
                    corrected['class_name'] = "Without Helmet"
                    corrected['class_id'] = 0
                else:
                    corrected['class_name'] = "With Helmet"
                    corrected['class_id'] = 1
                corrections.append(corrected)
                print(f"   ✏️ Corrected to: {corrected['class_name']}")
            elif choice == 'd':
                # Delete prediction - mark as false positive
                print(f"   🗑️ Marked as false positive")
                # Empty correction indicates deletion
                corrections.append({})
            else:
                print(f"   ⏭️ Skipped")
        
        # Ask for additional annotations
        print(f"\n➕ Do you want to add any missed detections?")
        add_choice = input("Add missed detections? (y/n): ").lower().strip()
        
        if add_choice == 'y':
            print(f"   📝 Note: Use the GUI interface for precise bounding box annotation")
            print(f"   Run: python -m app.feedback_interface")
        
        # Get user confidence
        try:
            confidence = float(input("\n🎯 Your confidence in this feedback (0.1-1.0): ") or "0.9")
            confidence = max(0.1, min(1.0, confidence))
        except ValueError:
            confidence = 0.9
        
        # Submit feedback
        self.rl_trainer.collect_feedback(
            image_path=image_path,
            original_predictions=predictions,
            corrected_annotations=corrections,
            user_confidence=confidence
        )
        
        print(f"✅ Feedback submitted! Confidence: {confidence:.1f}")
    
    def process_batch_with_feedback(self, folder_path: str, collect_feedback: bool = False):
        """Process a batch of images with optional feedback collection.
        
        Args:
            folder_path: Path to folder containing images
            collect_feedback: Whether to collect feedback for each image
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {folder_path}")
            return
        
        print(f"📁 Processing {len(image_files)} images from {folder_path}")
        
        results = []
        start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            try:
                result = self.detect_with_feedback_option(
                    str(image_file), 
                    save_result=True,
                    interactive=collect_feedback
                )
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
        
        # Summary
        total_time = time.time() - start_time
        total_detections = sum(r['total_detections'] for r in results)
        total_helmets = sum(r['helmet_count'] for r in results)
        total_no_helmets = sum(r['no_helmet_count'] for r in results)
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   - Images processed: {len(results)}")
        print(f"   - Total detections: {total_detections}")
        print(f"   - With helmet: {total_helmets}")
        print(f"   - Without helmet: {total_no_helmets}")
        print(f"   - Processing time: {total_time:.1f}s")
        print(f"   - Average per image: {total_time/len(results):.2f}s")
    
    def train_from_collected_feedback(self, epochs: int = 10):
        """Train the model using all collected feedback.
        
        Args:
            epochs: Number of training epochs
        """
        print(f"🏋️ Training model with collected feedback...")
        
        # Show current feedback stats
        stats = self.rl_trainer.get_training_stats()
        print(f"📊 Current feedback statistics:")
        for key, value in stats.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        
        if stats['pending_feedback'] == 0:
            print(f"⚠️ No new feedback to train on")
            return
        
        # Start training
        self.rl_trainer.train_from_feedback(epochs=epochs)
        
        # Reload the updated model
        self.detector = HelmetDetector(self.model_path)
        print(f"🔄 Model reloaded with improvements")
    
    def show_feedback_stats(self):
        """Display current feedback and training statistics."""
        stats = self.rl_trainer.get_training_stats()
        
        print(f"\n📊 TrafficAI Reinforcement Learning Statistics")
        print(f"=" * 50)
        print(f"Total Feedback Collected: {stats['total_feedback']}")
        print(f"Processed for Training: {stats['processed_feedback']}")
        print(f"Pending Training: {stats['pending_feedback']}")
        print(f"User Corrections: {stats['corrections']}")
        print(f"User Validations: {stats['validations']}")
        print(f"Average User Confidence: {stats['average_user_confidence']:.2f}")
        print(f"Replay Buffer Size: {stats['replay_buffer_size']}")
        
        if stats['pending_feedback'] > 0:
            print(f"\n💡 Recommendation: Run training to process {stats['pending_feedback']} pending feedback entries")
    
    def launch_gui_interface(self):
        """Launch the graphical feedback interface."""
        print(f"🖥️ Launching GUI feedback interface...")
        interface = FeedbackInterface()
        interface.run()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="TrafficAI - Helmet Detection with Reinforcement Learning")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--folder", type=str, help="Path to folder containing images")
    parser.add_argument("--model", type=str, default="models/helmet_yolov8.pt", help="Path to model file")
    parser.add_argument("--feedback", action="store_true", help="Collect interactive feedback")
    parser.add_argument("--train", action="store_true", help="Train model from collected feedback")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--stats", action="store_true", help="Show feedback statistics")
    parser.add_argument("--gui", action="store_true", help="Launch GUI feedback interface")
    
    args = parser.parse_args()
    
    # Initialize TrafficAI with RL
    try:
        traffic_ai = TrafficAIRL(args.model)
    except Exception as e:
        print(f"❌ Failed to initialize TrafficAI: {e}")
        return
    
    # Handle different modes
    if args.gui:
        traffic_ai.launch_gui_interface()
    
    elif args.stats:
        traffic_ai.show_feedback_stats()
    
    elif args.train:
        traffic_ai.train_from_collected_feedback(args.epochs)
    
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        traffic_ai.detect_with_feedback_option(args.image, interactive=args.feedback)
    
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        traffic_ai.process_batch_with_feedback(args.folder, args.feedback)
    
    else:
        # Show help and current stats
        print(f"🚀 TrafficAI - Helmet Detection with Reinforcement Learning")
        print(f"=" * 60)
        print(f"")
        print(f"Usage examples:")
        print(f"  🔍 Detect: python -m app.main_rl --image path/to/image.jpg")
        print(f"  📝 With feedback: python -m app.main_rl --image path/to/image.jpg --feedback")
        print(f"  📁 Process folder: python -m app.main_rl --folder path/to/images")
        print(f"  🏋️ Train model: python -m app.main_rl --train")
        print(f"  📊 Show stats: python -m app.main_rl --stats")
        print(f"  🖥️ GUI interface: python -m app.main_rl --gui")
        print(f"")
        
        traffic_ai.show_feedback_stats()


if __name__ == "__main__":
    main()
