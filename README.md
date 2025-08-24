# BatoNetra

Simple demo app that  
one detects riders without a helmet using a YOLOv8 model inspired by the Kaggle notebook  
two flags wrong lane presence using a per camera binary mask  
three extracts license plate text with YOLO for plate detection and Tesseract OCR

Reference for helmet model  
Kaggle notebook Helmet Detection using YOLOv8  
Place your trained weights as models/helmet_yolov8.pt

Reference for license plate model  
Place your plate detector weights as models/lp_yolov8.pt  
You can fine tune this model with the steps below

## Setup

Prerequisites  
Python 3 point 10 or newer  
A GPU is recommended to not end up like us

Install Python packages
```

pip install -r requirements.txt

```

Install system Tesseract OCR

On Ubuntu or Debian
```

sudo apt update
sudo apt install tesseract-ocr libtesseract-dev

```

On macOS with Homebrew
```

brew install tesseract

```

On Windows with Chocolatey
```

choco install tesseract

```

Verify Python libs
```

python -c "import ultralytics, cv2, numpy as np, pytesseract; print('ok')"

```

If Windows cannot find Tesseract add its path in your env or set it once in code  
Example path  
`C:\Program Files\Tesseract-OCR\tesseract.exe`

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

License plate model  
Use any public license plate dataset or your own  
Place the trained or downloaded weights at models/lp_yolov8.pt

## How it works

Helmet detection  
We run YOLOv8 on the image  
For each rider we check the helmet class  
If helmet class is not present on the rider head region we flag no_helmet

Wrong lane  
We compute a centroid for each rider or vehicle box  
If the centroid lies in a white pixel of the mask we flag wrong_lane

License plate extraction  
We detect license plates with a YOLOv8 model  
We crop the plate region  
We preprocess with grayscale and threshold and light denoise  
We send the crop to Tesseract to read the text  
We normalize the string to a plate like token and save it with the violation

## Run

Single image
```

python app/main.py --image data/samples/sample1.jpg

```

Batch folder
```

python app/main.py --folder data/samples

```

Enable license plate extraction in either mode
```

python app/main.py --image data/samples/sample1.jpg --plates

```

Specify custom weights if needed
```

python app/main.py --image data/samples/sample1.jpg --helmet-weights models/helmet\_yolov8.pt --plate-weights models/lp\_yolov8.pt

````

Output  
Console report and an annotated image saved next to the input  
When plates are enabled a CSV at outputs/plates.csv contains image name box id plate text and confidence

## File overview

app/main.py  
Entry point that loads the model runs inference applies the lane mask performs plate detection and OCR and saves results

app/inference.py  
Thin wrapper around YOLOv8 for helmet detection and vehicle detection if you add it

app/plates.py  
License plate detection and OCR utilities  
loads YOLO plate model  
preprocesses crops and calls Tesseract

app/lane.py  
Mask logic and helper to draw overlays

app/draw_masks.py  
Utility that lets you draft a mask from points if you prefer a quick script instead of an editor

models/helmet_yolov8.pt  
Weights file you provide

models/lp_yolov8.pt  
License plate detector weights file you provide

data/masks/wrong_lane_mask.png  
Binary mask for wrong lane region

## License plate OCR details

Tesseract setup in code  
If Tesseract is not on PATH set the binary path before calling OCR
```python
import pytesseract, os, platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
````

OCR call with common config

```python
import cv2
def ocr_plate(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    th  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    txt = pytesseract.image_to_string(th, config=config)
    return "".join(ch for ch in txt if ch.isalnum())
```

Typical pitfalls and tips

* Use `--psm 7` for a single line plate region
* Try `--psm 8` if characters are not aligned
* For low light increase contrast with CLAHE
* For skewed plates apply a min area rect and warp affine before OCR

## Fine tune the license plate detector

You can train or fine tune a small YOLOv8 model for plates

Dataset structure for Ultralytics

```
datasets/plates/
  images/
    train/
    val/
  labels/
    train/
    val/
plate.yaml
```

Example plate.yaml

```yaml
path: datasets/plates
train: images/train
val: images/val
names:
  0: license_plate
```

Start from a small model for speed

```
yolo detect train model=yolov8n.pt data=datasets/plates/plate.yaml imgsz=640 epochs=100 batch=16 workers=4 patience=25
```

Resume or continue

```
yolo detect train resume=True
```

Export the best weights
Copy `runs/detect/train/weights/best.pt` to `models/lp_yolov8.pt`

Validate

```
yolo detect val model=models/lp_yolov8.pt data=datasets/plates/plate.yaml imgsz=640
```

Inference sanity check

```
yolo detect predict model=models/lp_yolov8.pt source=data/samples save=True
```

Labeling tips

* Use a single class named license\_plate
* Label the full plate rectangle
* Include varied scales occlusion glare nighttime rain and different fonts
* Balance train and val sets and avoid leakage

## Database output

CSV at outputs/violations.csv
Columns
image path violation type bbox x1 y1 x2 y2 plate text plate score

You can point to a real database later with SQLite or Postgres
A simple SQLite writer can be added in app/db.py

## Notes

For a single image per camera this mask approach is very reliable
For moving video you can expand later with tracking and direction tests
You can add a vehicle detector to associate riders and plates with the correct vehicle for crowded scenes
