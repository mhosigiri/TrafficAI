# Helmet Detection Dataset Sources

## Kaggle Helmet Detection Dataset
**URL**: https://www.kaggle.com/datasets/andrewmvd/helmet-detection

**Instructions**:
  1. Go to: https://www.kaggle.com/datasets/andrewmvd/helmet-detection
  2. Click 'Download' (requires Kaggle account)
  3. Extract the zip file
  4. Copy images to data/images/
  5. Copy annotations to data/annotations/
  6. Run: python3 app/xml_to_yolo_converter.py

## Roboflow Helmet Detection
**URL**: https://universe.roboflow.com/search?q=helmet%20detection

**Instructions**:
  1. Browse datasets at: https://universe.roboflow.com
  2. Search for 'helmet detection'
  3. Choose a dataset and sign up
  4. Download in YOLOv8 format
  5. Extract to data/ folder

## Custom Dataset Collection
**URL**: manual

**Instructions**:
  1. Collect motorcycle/scooter images from traffic cameras
  2. Use labelImg tool to create annotations
  3. Save in Pascal VOC format (XML)
  4. Place images in data/images/
  5. Place XML files in data/annotations/
  6. Convert: python3 app/xml_to_yolo_converter.py

