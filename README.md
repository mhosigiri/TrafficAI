# README.md
cat > NepalTrafficLite/README.md << 'EOF'
# NepalTrafficLite

Simple demo app that
one detects riders without a helmet using a YOLOv8 model inspired by the Kaggle notebook
two flags wrong lane presence using a per camera binary mask

Reference for helmet model
Kaggle notebook Helmet Detection using YOLOv8
Place your trained weights as models/helmet_yolov8.pt

## Setup

Prerequisites
Python 3 point 10 or newer
A GPU is nice but not required

Install
pip install -r requirements.txt

Verify
python -c "import ultralytics, cv2, numpy as np; print('ok')"

## Prepare assets

Helmet model
Option A use the provided Kaggle training process then copy the best weights to models/helmet_yolov8.pt
Option B for a quick start you can also use a YOLOv8 nano model then fine tune later

Wrong lane mask
Create data/masks/wrong_lane_mask.png as a single channel image
White 255 means forbidden lane zone for this camera view
Black 0 means allowed
Tip use any image editor to paint the area

Samples
Put test images in data/samples
Example data/samples/sample1.jpg

## How it works

Helmet detection
We run YOLOv8 on the image
For each rider we check the helmet class
If helmet class is not present on the rider head region we flag no_helmet

Wrong lane
We compute a centroid for each rider or vehicle box
If the centroid lies in a white pixel of the mask we flag wrong_lane

## Run

Single image
python app/main.py --image data/samples/sample1.jpg

Batch folder
python app/main.py --folder data/samples

Output
Console report and an annotated image saved next to the input

## File overview

app/main.py
Entry point that loads the model runs inference applies the lane mask and saves results

app/inference.py
Thin wrapper around YOLOv8 for helmet detection

app/lane.py
Mask logic and helper to draw overlays

app/draw_masks.py
Utility that lets you draft a mask from points if you prefer a quick script instead of an editor

models/helmet_yolov8.pt
Weights file you provide

data/masks/wrong_lane_mask.png
Binary mask for wrong lane region

## Notes

For a single image per camera this mask approach is very reliable
For moving video you can expand later with tracking and direction tests

EOF
