import os
import cv2
import urllib.request

# Model file paths
MODEL_DIR = "models"
MODEL_CFG = os.path.join(MODEL_DIR, "yolov3.cfg")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "yolov3.weights")
MODEL_NAMES = os.path.join(MODEL_DIR, "coco.names")

# URLs for downloading models
CFG_URL = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
NAMES_URL = "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"

# Function to download the model if missing
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_CFG):
        print("Downloading YOLOv3 config...")
        urllib.request.urlretrieve(CFG_URL, MODEL_CFG)

    if not os.path.exists(MODEL_WEIGHTS):
        print("Downloading YOLOv3 weights (this may take time)...")
        urllib.request.urlretrieve(WEIGHTS_URL, MODEL_WEIGHTS)

    if not os.path.exists(MODEL_NAMES):
        print("Downloading COCO class names...")
        urllib.request.urlretrieve(NAMES_URL, MODEL_NAMES)

# Download model before loading
download_model()

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
