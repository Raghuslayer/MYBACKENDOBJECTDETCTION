function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    if (fileInput.files.length === 0) {
        alert("Please select an image!");
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput.files[0]);

    fetch("https://yolo-backend.onrender.com/detect", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("resultImage").src = "https://yolo-backend.onrender.com" + data.processed_image_url;
    })
    .catch(error => console.error("Error:", error));
}
