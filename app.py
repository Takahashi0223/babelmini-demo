import os
import cv2
import numpy as np
import base64
import json
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from io import BytesIO

app = Flask(__name__)

def load_metadata():
    with open("sample_book_spines/book_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

def decode_image(data_url):
    header, encoded = data_url.split(",", 1)
    binary_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(binary_data)).convert("RGB")
    return np.array(image)

def compare_images(uploaded_img, reference_img):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(uploaded_img, None)
    kp2, des2 = orb.detectAndCompute(reference_img, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    uploaded_img = decode_image(data["image"])
    uploaded_img = cv2.cvtColor(uploaded_img, cv2.COLOR_RGB2GRAY)

    metadata = load_metadata()
    best_match = None
    best_score = -1

    for filename in metadata:
        path = os.path.join("sample_book_spines", filename)
        ref_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        score = compare_images(uploaded_img, ref_img)
        if score > best_score:
            best_score = score
            best_match = filename

    result = metadata.get(best_match, {"title": "不明", "author": "不明"})
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
