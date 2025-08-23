
# Install roboflow if you want to use their datasets
# pip install roboflow

from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="YOUR_API_KEY")

# Download a helmet detection dataset
project = rf.workspace("your-workspace").project("helmet-detection")
dataset = project.version(1).download("yolov8")
