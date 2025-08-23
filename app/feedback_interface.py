"""
User Feedback Interface for Helmet Detection
Allows users to provide feedback on model predictions for reinforcement learning.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

from .helmet_detector import HelmetDetector
from .rl_trainer import RLHelmetTrainer


class FeedbackInterface:
    """GUI interface for collecting user feedback on helmet detection results."""
    
    def __init__(self):
        """Initialize the feedback interface."""
        self.root = tk.Tk()
        self.root.title("TrafficAI - Helmet Detection Feedback")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.detector = HelmetDetector()
        self.rl_trainer = RLHelmetTrainer()
        
        # Current image and predictions
        self.current_image_path = None
        self.current_image = None
        self.current_predictions = []
        self.corrected_annotations = []
        self.selected_bbox = None
        self.drawing_bbox = False
        self.bbox_start = None
        
        # GUI components
        self.setup_gui()
        
        print("🖥️ Feedback Interface initialized")
    
    def setup_gui(self):
        """Setup the GUI components."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image upload button
        upload_btn = ttk.Button(left_frame, text="📁 Upload Image", command=self.upload_image)
        upload_btn.pack(pady=5)
        
        # Image canvas
        self.canvas = tk.Canvas(left_frame, bg="white", width=600, height=400)
        self.canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Control buttons
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(pady=5)
        
        ttk.Button(controls_frame, text="🔍 Detect Helmets", command=self.detect_helmets).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="📝 Submit Feedback", command=self.submit_feedback).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🧹 Clear All", command=self.clear_annotations).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Predictions and feedback
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Predictions section
        ttk.Label(right_frame, text="🤖 Model Predictions", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        
        self.predictions_tree = ttk.Treeview(right_frame, columns=("Class", "Confidence", "Bbox"), show="headings", height=6)
        self.predictions_tree.heading("Class", text="Class")
        self.predictions_tree.heading("Confidence", text="Confidence")
        self.predictions_tree.heading("Bbox", text="Bounding Box")
        self.predictions_tree.column("Class", width=100)
        self.predictions_tree.column("Confidence", width=80)
        self.predictions_tree.column("Bbox", width=120)
        self.predictions_tree.pack(pady=5, fill=tk.X)
        self.predictions_tree.bind("<Button-1>", self.on_prediction_select)
        
        # Feedback section
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(right_frame, text="✏️ Your Corrections", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        
        # Annotation tools
        tools_frame = ttk.Frame(right_frame)
        tools_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(tools_frame, text="Add:").pack(side=tk.LEFT)
        ttk.Button(tools_frame, text="➕ With Helmet", command=lambda: self.set_annotation_mode("With Helmet")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools_frame, text="➖ Without Helmet", command=lambda: self.set_annotation_mode("Without Helmet")).pack(side=tk.LEFT, padx=2)
        
        self.annotation_mode = None
        self.mode_label = ttk.Label(right_frame, text="Mode: Select", foreground="blue")
        self.mode_label.pack(pady=5)
        
        # Corrections tree
        self.corrections_tree = ttk.Treeview(right_frame, columns=("Class", "Bbox", "Action"), show="headings", height=6)
        self.corrections_tree.heading("Class", text="Class")
        self.corrections_tree.heading("Bbox", text="Bounding Box")
        self.corrections_tree.heading("Action", text="Action")
        self.corrections_tree.column("Class", width=100)
        self.corrections_tree.column("Bbox", width=120)
        self.corrections_tree.column("Action", width=80)
        self.corrections_tree.pack(pady=5, fill=tk.X)
        
        # User confidence
        confidence_frame = ttk.Frame(right_frame)
        confidence_frame.pack(fill=tk.X, pady=10)
        ttk.Label(confidence_frame, text="Your Confidence:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.9)
        self.confidence_scale = ttk.Scale(confidence_frame, from_=0.1, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL)
        self.confidence_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Feedback stats
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(right_frame, text="📊 Training Stats", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        
        self.stats_text = tk.Text(right_frame, height=8, width=40)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Train button
        ttk.Button(right_frame, text="🏋️ Train from Feedback", command=self.train_model).pack(pady=10, fill=tk.X)
        
        # Update stats initially
        self.update_stats()
    
    def upload_image(self):
        """Upload and display an image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.load_image(file_path)
            self.clear_annotations()
            print(f"📁 Loaded image: {file_path}")
    
    def load_image(self, image_path: str):
        """Load and display image on canvas."""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image.copy()
        
        # Resize for display
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
            h, w = image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            image_resized = cv2.resize(image, (new_w, new_h))
            self.display_scale = scale
            
            # Convert to PhotoImage
            image_pil = Image.fromarray(image_resized)
            self.photo = ImageTk.PhotoImage(image_pil)
            
            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
    
    def detect_helmets(self):
        """Run helmet detection on current image."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        print("🔍 Running helmet detection...")
        self.current_predictions = self.detector.detect(self.current_image_path)
        self.update_predictions_display()
        self.draw_predictions()
        print(f"✅ Found {len(self.current_predictions)} detections")
    
    def update_predictions_display(self):
        """Update the predictions tree view."""
        # Clear existing items
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Add predictions
        for i, pred in enumerate(self.current_predictions):
            bbox_str = f"({pred['bbox_xyxy'][0]:.0f},{pred['bbox_xyxy'][1]:.0f})"
            self.predictions_tree.insert("", "end", values=(
                pred['class_name'],
                f"{pred['confidence']:.2f}",
                bbox_str
            ))
    
    def draw_predictions(self):
        """Draw prediction bounding boxes on canvas."""
        if not hasattr(self, 'display_scale'):
            return
        
        # Draw original predictions in blue
        for pred in self.current_predictions:
            bbox = pred['bbox_xyxy']
            x1, y1, x2, y2 = [coord * self.display_scale for coord in bbox]
            
            # Adjust for canvas center positioning
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width = self.current_image.shape[1] * self.display_scale
            img_height = self.current_image.shape[0] * self.display_scale
            
            offset_x = (canvas_width - img_width) // 2
            offset_y = (canvas_height - img_height) // 2
            
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            
            color = "blue" if pred.get('is_helmet') else "red"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="prediction")
            self.canvas.create_text(x1, y1-10, text=f"{pred['class_name']}: {pred['confidence']:.2f}", 
                                  anchor="sw", fill=color, tags="prediction")
        
        # Draw corrected annotations in green
        for correction in self.corrected_annotations:
            if 'bbox_xyxy' in correction:
                bbox = correction['bbox_xyxy']
                x1, y1, x2, y2 = [coord * self.display_scale for coord in bbox]
                
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y
                
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=3, tags="correction")
                self.canvas.create_text(x1, y1-10, text=f"✓ {correction['class_name']}", 
                                      anchor="sw", fill="green", tags="correction")
    
    def set_annotation_mode(self, class_name: str):
        """Set the annotation mode for drawing new bboxes."""
        self.annotation_mode = class_name
        self.mode_label.config(text=f"Mode: Drawing {class_name}", foreground="green")
        print(f"🎯 Annotation mode: {class_name}")
    
    def on_canvas_click(self, event):
        """Handle canvas click events."""
        if self.annotation_mode:
            self.drawing_bbox = True
            self.bbox_start = (event.x, event.y)
    
    def on_canvas_drag(self, event):
        """Handle canvas drag events."""
        if self.drawing_bbox and self.bbox_start:
            # Clear previous temp rectangle
            self.canvas.delete("temp_bbox")
            
            # Draw temporary rectangle
            x1, y1 = self.bbox_start
            x2, y2 = event.x, event.y
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="orange", width=2, tags="temp_bbox")
    
    def on_canvas_release(self, event):
        """Handle canvas release events."""
        if self.drawing_bbox and self.bbox_start:
            self.drawing_bbox = False
            
            # Convert canvas coordinates to image coordinates
            x1, y1 = self.bbox_start
            x2, y2 = event.x, event.y
            
            # Ensure x1,y1 is top-left
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Convert to image coordinates
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width = self.current_image.shape[1] * self.display_scale
            img_height = self.current_image.shape[0] * self.display_scale
            
            offset_x = (canvas_width - img_width) // 2
            offset_y = (canvas_height - img_height) // 2
            
            img_x1 = (x1 - offset_x) / self.display_scale
            img_y1 = (y1 - offset_y) / self.display_scale
            img_x2 = (x2 - offset_x) / self.display_scale
            img_y2 = (y2 - offset_y) / self.display_scale
            
            # Validate bbox is within image bounds
            if (img_x1 >= 0 and img_y1 >= 0 and 
                img_x2 <= self.current_image.shape[1] and 
                img_y2 <= self.current_image.shape[0] and
                abs(img_x2 - img_x1) > 10 and abs(img_y2 - img_y1) > 10):
                
                # Add correction
                correction = {
                    'class_name': self.annotation_mode,
                    'class_id': 1 if self.annotation_mode == "With Helmet" else 0,
                    'bbox_xyxy': [img_x1, img_y1, img_x2, img_y2],
                    'confidence': 1.0,
                    'action': 'added'
                }
                
                self.corrected_annotations.append(correction)
                self.update_corrections_display()
                
                print(f"➕ Added correction: {self.annotation_mode}")
            
            # Clear temp drawing
            self.canvas.delete("temp_bbox")
            self.bbox_start = None
            
            # Reset mode
            self.annotation_mode = None
            self.mode_label.config(text="Mode: Select", foreground="blue")
            
            # Redraw
            self.draw_predictions()
    
    def on_prediction_select(self, event):
        """Handle prediction selection."""
        selection = self.predictions_tree.selection()
        if selection:
            item = self.predictions_tree.item(selection[0])
            print(f"📍 Selected prediction: {item['values']}")
    
    def update_corrections_display(self):
        """Update the corrections tree view."""
        # Clear existing items
        for item in self.corrections_tree.get_children():
            self.corrections_tree.delete(item)
        
        # Add corrections
        for correction in self.corrected_annotations:
            bbox_str = f"({correction['bbox_xyxy'][0]:.0f},{correction['bbox_xyxy'][1]:.0f})"
            self.corrections_tree.insert("", "end", values=(
                correction['class_name'],
                bbox_str,
                correction.get('action', 'modified')
            ))
    
    def clear_annotations(self):
        """Clear all annotations and predictions."""
        self.current_predictions = []
        self.corrected_annotations = []
        self.annotation_mode = None
        self.mode_label.config(text="Mode: Select", foreground="blue")
        
        # Clear displays
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        for item in self.corrections_tree.get_children():
            self.corrections_tree.delete(item)
        
        # Clear canvas
        self.canvas.delete("prediction")
        self.canvas.delete("correction")
        self.canvas.delete("temp_bbox")
        
        print("🧹 Cleared all annotations")
    
    def submit_feedback(self):
        """Submit user feedback for training."""
        if not self.current_image_path or not self.current_predictions:
            messagebox.showwarning("Warning", "Please upload an image and run detection first")
            return
        
        try:
            # Get user confidence
            confidence = self.confidence_var.get()
            
            # Submit feedback
            self.rl_trainer.collect_feedback(
                image_path=self.current_image_path,
                original_predictions=self.current_predictions,
                corrected_annotations=self.corrected_annotations,
                user_confidence=confidence
            )
            
            messagebox.showinfo("Success", f"Feedback submitted successfully!\nConfidence: {confidence:.1f}")
            print(f"📝 Feedback submitted with confidence {confidence:.2f}")
            
            # Update stats
            self.update_stats()
            
            # Clear for next image
            self.clear_annotations()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to submit feedback: {e}")
            print(f"❌ Error submitting feedback: {e}")
    
    def train_model(self):
        """Train the model using collected feedback."""
        try:
            # Show confirmation dialog
            result = messagebox.askyesno(
                "Confirm Training", 
                "Start training with collected feedback?\nThis may take several minutes."
            )
            
            if not result:
                return
            
            # Run training in a separate thread to avoid blocking GUI
            def train_thread():
                try:
                    print("🏋️ Starting RL training...")
                    self.rl_trainer.train_from_feedback(epochs=10, min_feedback_count=3)
                    
                    # Update GUI in main thread
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Training completed successfully!"))
                    self.root.after(0, self.update_stats)
                    
                except Exception as e:
                    error_msg = f"Training failed: {e}"
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    print(f"❌ {error_msg}")
            
            threading.Thread(target=train_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def update_stats(self):
        """Update the training statistics display."""
        try:
            stats = self.rl_trainer.get_training_stats()
            
            stats_text = "📊 Training Statistics:\n\n"
            stats_text += f"Total Feedback: {stats['total_feedback']}\n"
            stats_text += f"Processed: {stats['processed_feedback']}\n"
            stats_text += f"Pending: {stats['pending_feedback']}\n"
            stats_text += f"Corrections: {stats['corrections']}\n"
            stats_text += f"Validations: {stats['validations']}\n"
            stats_text += f"Avg Confidence: {stats['average_user_confidence']:.2f}\n"
            stats_text += f"Buffer Size: {stats['replay_buffer_size']}\n\n"
            
            # Add instructions
            stats_text += "💡 Instructions:\n"
            stats_text += "1. Upload an image\n"
            stats_text += "2. Click 'Detect Helmets'\n"
            stats_text += "3. Draw corrections if needed\n"
            stats_text += "4. Submit feedback\n"
            stats_text += "5. Train when ready"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
        except Exception as e:
            print(f"⚠️ Error updating stats: {e}")
    
    def run(self):
        """Run the feedback interface."""
        print("🚀 Starting TrafficAI Feedback Interface")
        print("Upload images, provide feedback, and train the model!")
        self.root.mainloop()


def main():
    """Main function to run the feedback interface."""
    interface = FeedbackInterface()
    interface.run()


if __name__ == "__main__":
    main()
