from app import app
from flask import render_template, request, redirect, jsonify, make_response
import os
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from app import ocr_scan
from app import surviviol_model
import time

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

def allowed_image_ext(filename):
    if not "." in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGES_EXTENSIONS"]:
        return True
    else:
        return False

def allowed_image_size(filesize):
    if int(filesize) <= app.config["MAX_IMAGE_SIZE"]:
        return True
    else:
        return False

# Phase-1: Uploading images
@app.route('/phase-one', methods=["GET", "POST"])
def upload_image():
    upload_status = ""
    access_path = ""
    if request.method=="POST":
        if request.files: 
            image = request.files['croppedImage']
            
            filename = secure_filename(image.filename)
            path = os.path.join(app.config["IMAGE_UPLOAD_PATH"], filename)
            image.save(path)

            access_path = os.path.join(app.config["IMAGE_ACCESS_PATH"], filename)
            
            print('Image has been uploaded.')
            upload_status = "Image has been uploaded."
            return render_template('index.html', image_upload_status=upload_status, img_path=access_path) # objects unused yet. 
        else:
            upload_status = "Image has failed to upload."
            return render_template('index.html', image_upload_status=upload_status)

    return render_template('index.html', image_upload_status=upload_status, img_path=access_path) # objects unused yet.

# Phase-2: Scan text from uploaded image
@app.route('/text-scan', methods=["POST"])
def text_scan():
    req = request.get_json()

    img = req['filename']
    print(img)

    text = ocr_scan.get_text(os.path.join(app.config["IMAGE_UPLOAD_PATH"], img))

    resp = make_response(jsonify({"text": text}), 200)
    return resp

# Phase-3: Call saved model to predict
@app.route('/predict', methods=["POST"])
def predict():
    req = request.get_json()

    text = req['text']
    print('Text will be predicted:', text)

    prediction = surviviol_model.predict(text)
    
    resp = make_response(jsonify({"prediction": prediction}), 200)
    # time.sleep(3)
    return resp