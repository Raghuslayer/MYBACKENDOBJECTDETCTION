from flask import Flask, request, send_file, render_template
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load the YOLO model (pre-trained YOLOv8)
model = YOLO("yolov8n.pt")  # 'n' for nano (lightweight), can use 'm', 'l', etc. for larger models

# Ensure an uploads folder exists
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # Serve the HTML frontend
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # 1. Receive the image from the frontend
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No image selected", 400

    # Save the uploaded image
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # 2. Process the image with YOLO
    results = model(input_path)  # Run YOLO inference
    labeled_image = results[0].plot()  # Get the labeled image (as numpy array)

    # Save the labeled image
    output_filename = "labeled_" + file.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, labeled_image)  # Save as an image file

    # 3. Send the labeled image back to the frontend
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    # Run the Flask app continuously
    app.run(host='0.0.0.0', port=5000, debug=True)
