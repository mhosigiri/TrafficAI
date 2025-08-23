"""
Reinforcement Learning Trainer for Helmet Detection
Implements online learning from user feedback to improve model accuracy.
"""

import json
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import pickle
import cv2


class FeedbackDatabase:
    """Database to store user feedback and training examples."""
    
    def __init__(self, db_path: str = "data/feedback.db"):
        """Initialize the feedback database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                original_predictions TEXT,  -- JSON string of model predictions
                corrected_annotations TEXT, -- JSON string of user corrections
                feedback_type TEXT,         -- 'correction', 'validation', 'new_annotation'
                confidence_score REAL,     -- User confidence in their feedback
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,        -- Version of model that made prediction
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Training metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_session TEXT,
                epoch INTEGER,
                loss REAL,
                accuracy REAL,
                feedback_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, image_path: str, original_predictions: List[Dict], 
                    corrected_annotations: List[Dict], feedback_type: str,
                    confidence_score: float = 1.0, model_version: str = "v1.0"):
        """Add user feedback to database.
        
        Args:
            image_path: Path to the image
            original_predictions: Model's original predictions
            corrected_annotations: User's corrected annotations
            feedback_type: Type of feedback ('correction', 'validation', etc.)
            confidence_score: User's confidence in their feedback (0-1)
            model_version: Version of the model that made the prediction
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (image_path, original_predictions, corrected_annotations, 
             feedback_type, confidence_score, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            image_path,
            json.dumps(original_predictions),
            json.dumps(corrected_annotations),
            feedback_type,
            confidence_score,
            model_version
        ))
        
        conn.commit()
        conn.close()
    
    def get_unprocessed_feedback(self, limit: int = None) -> List[Dict]:
        """Get unprocessed feedback for training.
        
        Args:
            limit: Maximum number of feedback entries to return
            
        Returns:
            List of feedback dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT id, image_path, original_predictions, corrected_annotations,
                   feedback_type, confidence_score, model_version, timestamp
            FROM feedback 
            WHERE processed = FALSE
            ORDER BY timestamp DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        feedback_list = []
        for row in rows:
            feedback_list.append({
                'id': row[0],
                'image_path': row[1],
                'original_predictions': json.loads(row[2]),
                'corrected_annotations': json.loads(row[3]),
                'feedback_type': row[4],
                'confidence_score': row[5],
                'model_version': row[6],
                'timestamp': row[7]
            })
        
        return feedback_list
    
    def mark_feedback_processed(self, feedback_ids: List[int]):
        """Mark feedback as processed.
        
        Args:
            feedback_ids: List of feedback IDs to mark as processed
        """
        if not feedback_ids:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(feedback_ids))
        cursor.execute(f'''
            UPDATE feedback 
            SET processed = TRUE 
            WHERE id IN ({placeholders})
        ''', feedback_ids)
        
        conn.commit()
        conn.close()


class RewardCalculator:
    """Calculate rewards for reinforcement learning based on user feedback."""
    
    def __init__(self):
        """Initialize reward calculator."""
        self.reward_weights = {
            'correct_detection': 1.0,       # Reward for correct detection
            'missed_detection': -0.8,       # Penalty for missing an object
            'false_positive': -0.6,         # Penalty for false detection
            'bbox_accuracy': 0.5,           # Reward for accurate bounding box
            'confidence_bonus': 0.2,        # Bonus based on user confidence
            'class_accuracy': 0.3           # Reward for correct classification
        }
    
    def calculate_reward(self, original_pred: Dict, corrected_anno: Dict, 
                        user_confidence: float = 1.0) -> float:
        """Calculate reward based on prediction vs correction.
        
        Args:
            original_pred: Original model prediction
            corrected_anno: User's corrected annotation
            user_confidence: User's confidence in their correction (0-1)
            
        Returns:
            Reward score (can be positive or negative)
        """
        reward = 0.0
        
        # If user provided correction, it means original was wrong
        if corrected_anno:
            # Check if it was a missed detection (no original prediction)
            if not original_pred:
                reward += self.reward_weights['missed_detection']
            else:
                # Check bbox accuracy using IoU
                iou = self._calculate_iou(
                    original_pred.get('bbox_xyxy', []),
                    corrected_anno.get('bbox_xyxy', [])
                )
                
                # Reward based on IoU
                if iou > 0.7:
                    reward += self.reward_weights['bbox_accuracy'] * iou
                elif iou > 0.3:
                    reward += self.reward_weights['bbox_accuracy'] * iou * 0.5
                else:
                    reward += self.reward_weights['false_positive']
                
                # Check class accuracy
                if (original_pred.get('class_name') == corrected_anno.get('class_name')):
                    reward += self.reward_weights['class_accuracy']
                else:
                    reward -= self.reward_weights['class_accuracy']
        
        else:
            # No correction means original prediction was good
            reward += self.reward_weights['correct_detection']
        
        # Apply confidence bonus/penalty
        confidence_modifier = user_confidence * self.reward_weights['confidence_bonus']
        reward += confidence_modifier
        
        return reward
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1)
        """
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class RLHelmetTrainer:
    """Reinforcement Learning trainer for helmet detection model."""
    
    def __init__(self, model_path: str = "models/helmet_yolov8.pt", 
                 device: str = None):
        """Initialize RL trainer.
        
        Args:
            model_path: Path to the base helmet detection model
            device: Device to use for training ('cuda', 'mps', 'cpu')
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.db = FeedbackDatabase()
        self.reward_calc = RewardCalculator()
        
        # Load base model
        self.model = YOLO(model_path)
        
        # RL specific parameters
        self.learning_rate = 0.001
        self.batch_size = 8
        self.replay_buffer_size = 1000
        self.replay_buffer = []
        
        # Training history
        self.training_history = []
        
        print(f"🤖 RL Trainer initialized on {self.device.upper()}")
    
    def _get_device(self, device: str = None) -> str:
        """Get the best available device."""
        if device:
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def collect_feedback(self, image_path: str, original_predictions: List[Dict],
                        corrected_annotations: List[Dict], user_confidence: float = 1.0):
        """Collect user feedback for an image.
        
        Args:
            image_path: Path to the image
            original_predictions: Model's original predictions
            corrected_annotations: User's corrected annotations
            user_confidence: User's confidence in their feedback (0-1)
        """
        # Determine feedback type
        if corrected_annotations:
            feedback_type = "correction"
        else:
            feedback_type = "validation"
        
        # Store in database
        self.db.add_feedback(
            image_path=image_path,
            original_predictions=original_predictions,
            corrected_annotations=corrected_annotations,
            feedback_type=feedback_type,
            confidence_score=user_confidence,
            model_version="rl_v1.0"
        )
        
        # Calculate rewards and add to replay buffer
        for i, orig_pred in enumerate(original_predictions):
            corrected = corrected_annotations[i] if i < len(corrected_annotations) else {}
            reward = self.reward_calc.calculate_reward(orig_pred, corrected, user_confidence)
            
            experience = {
                'image_path': image_path,
                'prediction': orig_pred,
                'correction': corrected,
                'reward': reward,
                'user_confidence': user_confidence
            }
            
            self.replay_buffer.append(experience)
            
            # Maintain buffer size
            if len(self.replay_buffer) > self.replay_buffer_size:
                self.replay_buffer.pop(0)
        
        print(f"📝 Collected feedback for {image_path} with {len(corrected_annotations)} corrections")
    
    def train_from_feedback(self, epochs: int = 10, min_feedback_count: int = 5):
        """Train the model using collected feedback.
        
        Args:
            epochs: Number of training epochs
            min_feedback_count: Minimum feedback count before training
        """
        # Get unprocessed feedback
        feedback_data = self.db.get_unprocessed_feedback()
        
        if len(feedback_data) < min_feedback_count:
            print(f"⚠️ Not enough feedback ({len(feedback_data)}) for training. Need at least {min_feedback_count}.")
            return
        
        print(f"🏋️ Starting RL training with {len(feedback_data)} feedback examples...")
        print(f"📊 Training parameters:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Device: {self.device.upper()}")
        print(f"   - Learning rate: {self.learning_rate}")
        
        # Create training dataset from feedback
        training_images = []
        training_labels = []
        
        for feedback in feedback_data:
            image_path = feedback['image_path']
            if Path(image_path).exists():
                # Use corrected annotations as ground truth
                corrected_annos = feedback['corrected_annotations']
                if corrected_annos:
                    training_images.append(image_path)
                    training_labels.append(corrected_annos)
        
        if not training_images:
            print("❌ No valid training images found in feedback")
            return
        
        # Create temporary dataset configuration
        temp_dataset_config = self._create_temp_dataset(training_images, training_labels)
        
        try:
            # Fine-tune the model with feedback data
            print(f"🔧 Fine-tuning model with {len(training_images)} corrected examples...")
            
            results = self.model.train(
                data=temp_dataset_config,
                epochs=epochs,
                imgsz=640,
                batch=self.batch_size,
                name='rl_training',
                patience=5,
                save=True,
                device=self.device,
                verbose=True,
                lr0=self.learning_rate,  # Lower learning rate for fine-tuning
                warmup_epochs=1,
                # Keep most weights frozen, only fine-tune detection head
                freeze=10  # Freeze first 10 layers
            )
            
            # Update model path to the newly trained model
            best_model_path = "runs/detect/rl_training/weights/best.pt"
            if Path(best_model_path).exists():
                self.model = YOLO(best_model_path)
                
                # Copy to main model location
                import shutil
                shutil.copy2(best_model_path, self.model_path)
                print(f"✅ Updated model saved to {self.model_path}")
            
            # Mark feedback as processed
            feedback_ids = [f['id'] for f in feedback_data]
            self.db.mark_feedback_processed(feedback_ids)
            
            print(f"✅ RL Training completed successfully!")
            
        except Exception as e:
            print(f"❌ RL Training failed: {e}")
        
        finally:
            # Cleanup temporary files
            if Path(temp_dataset_config).exists():
                Path(temp_dataset_config).unlink()
    
    def _create_temp_dataset(self, images: List[str], labels: List[List[Dict]]) -> str:
        """Create temporary dataset configuration for training.
        
        Args:
            images: List of image paths
            labels: List of label dictionaries
            
        Returns:
            Path to temporary dataset configuration
        """
        # Create temporary directory
        temp_dir = Path("temp_rl_dataset")
        temp_dir.mkdir(exist_ok=True)
        
        # Create YOLO format labels
        labels_dir = temp_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        for img_path, img_labels in zip(images, labels):
            img_name = Path(img_path).stem
            label_file = labels_dir / f"{img_name}.txt"
            
            # Convert to YOLO format
            with open(label_file, 'w') as f:
                for label in img_labels:
                    if 'bbox' in label and 'class_id' in label:
                        # Normalize bbox coordinates
                        x_center, y_center, width, height = label['bbox']
                        class_id = label['class_id']
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Create dataset configuration
        dataset_config = temp_dir / "dataset.yaml"
        with open(dataset_config, 'w') as f:
            f.write(f"""
train: {Path().absolute() / "data" / "images"}
val: {Path().absolute() / "data" / "images"}

nc: 2
names:
  0: Without Helmet
  1: With Helmet
""")
        
        return str(dataset_config)
    
    def get_training_stats(self) -> Dict:
        """Get training statistics and feedback summary.
        
        Returns:
            Dictionary with training statistics
        """
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get feedback counts
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE processed = TRUE")
        processed_feedback = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'correction'")
        corrections = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence_score) FROM feedback")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'processed_feedback': processed_feedback,
            'pending_feedback': total_feedback - processed_feedback,
            'corrections': corrections,
            'validations': total_feedback - corrections,
            'average_user_confidence': round(avg_confidence, 3),
            'replay_buffer_size': len(self.replay_buffer)
        }


def main():
    """Example usage of the RL trainer."""
    print("🚀 TrafficAI Reinforcement Learning Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = RLHelmetTrainer()
    
    # Show current stats
    stats = trainer.get_training_stats()
    print("📊 Current Statistics:")
    for key, value in stats.items():
        print(f"   - {key.replace('_', ' ').title()}: {value}")
    
    print("\n💡 Usage Examples:")
    print("1. Collect feedback: trainer.collect_feedback(image_path, predictions, corrections)")
    print("2. Train from feedback: trainer.train_from_feedback(epochs=10)")
    print("3. Get stats: trainer.get_training_stats()")


if __name__ == "__main__":
    main()
