# AGENTS.md
cat > NepalTrafficLite/AGENTS.md << 'EOF'
# NepalTrafficLite Agents

Role 1 Computer Vision Agent
Purpose
Detect riders without a helmet using a YOLOv8 model from the Kaggle reference
Output a list of rider boxes with helmet boolean and confidence

Inputs
Image in BGR or RGB
Model weights at models/helmet_yolov8.pt

Outputs
JSON style dict
  detections
    bbox x y w h
    is_helmet true or false
    conf float

Role 2 Lane Rule Agent
Purpose
Flag wrong lane position using a simple binary mask for each camera view
The mask marks forbidden lane area for the observed direction

Inputs
Image
Binary mask at data/masks/wrong_lane_mask.png where white means forbidden
Object points to check typically vehicle centroids or rider centroids

Outputs
JSON style dict
  violations
    type wrong_lane
    point x y
    evidence mask_hit true

Coordinator
Purpose
Run both agents on each input image
Unify results and print a compact report
EOF
