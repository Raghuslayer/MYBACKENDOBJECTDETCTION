import os
import cv2
import urllib.request
import numpy as np

# Model file paths
MODEL_DIR = "models"
MODEL_CFG = os.path.join(MODEL_DIR, "yolov3.cfg")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "yolov3.weights")
MODEL_NAMES = os.path.join(MODEL_DIR, "coco.names")

# URLs for downloading models
CFG_URL = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg"
WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
NAMES_URL = "https://github.com/pjreddie/darknet/raw/master/data/coco.names"

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

# Load class names
with open(MODEL_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection
def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Convert to blob
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Forward pass
    outs = net.forward(output_layers)

    # Process detection results
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw labels and bounding boxes
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save labeled image
    output_path = os.path.join("output", os.path.basename(image_path))
    if not os.path.exists("output"):
        os.makedirs("output")
    cv2.imwrite(output_path, image)
    
    return output_path
