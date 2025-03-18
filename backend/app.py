from flask import Flask, request, jsonify, send_file
from detect import detect_objects  # Ensure detect.py has this function
import os

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Perform object detection
    try:
        output_image_path = detect_objects(image_path)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    # Return the processed image
    return send_file(output_image_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
