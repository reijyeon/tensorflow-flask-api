from flask import Flask, request, jsonify, send_file
import werkzeug

import os
import time
import tensorflow_hub as hub
import tensorflow as tf
from functions import preprocess_image, save_image, plot_image, downscale_image
#import matplotlib.pyplot as plt

app = Flask(__name__)
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

def esrgan(inputImage):
    SAVED_MODEL_PATH = "./model"
    IMAGE_PATH = inputImage
    hr_image = preprocess_image(IMAGE_PATH)

    lr_image = downscale_image(tf.squeeze(hr_image))

    # Plotting Low Resolution Image
    plot_image(tf.squeeze(lr_image), title="Low_Resolution")

    model = hub.load(SAVED_MODEL_PATH)

    start = time.time()
    fake_image = model(lr_image)
    fake_image = tf.squeeze(fake_image)
    print("Time Taken: %f" % (time.time() - start))

    plot_image(tf.squeeze(fake_image), title="Super_Resolution")
    save_image(tf.squeeze(fake_image), filename="Super_Resolution")
    # Calculating PSNR wrt Original Image
    psnr = tf.image.psnr(
        tf.clip_by_value(fake_image, 0, 255),
        tf.clip_by_value(hr_image, 0, 255), max_val=255)
    print("PSNR Achieved: %f" % psnr)

def superResolution(inputImage, filename):
# Declaring Constants
    IMAGE_PATH = inputImage
    SAVED_MODEL_PATH = "./model"

    hr_image = preprocess_image(IMAGE_PATH)

    # Plotting Original Resolution image
    plot_image(tf.squeeze(hr_image), title="SuperOriginal_Image")
    save_image(tf.squeeze(hr_image), filename="SuperOriginal_Image")

    model = hub.load(SAVED_MODEL_PATH)

    start = time.time()
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    print("Time Taken: %f" % (time.time() - start))
    
    # Plotting Super Resolution Image
    plot_image(tf.squeeze(fake_image), title="Upscaled_%s" % filename)
    save_image(tf.squeeze(fake_image), filename="Upscaled_%s" % filename)

@app.route('/upload', methods=['POST'])
def updload():
    if(request.method == 'POST'):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadimages/" + filename+".jpg")

        inputImage = "./uploadimages/"+filename+".jpg"

        esrgan(inputImage)
        superResolution(inputImage,filename)

        outputImage = "Upscaled_%s" % filename+".jpg"
        
        return jsonify({
            "result": outputImage
        })
        
@app.route('/download/<fname>', methods=['GET'])
def download(fname):
    return send_file(fname, as_attachment=True)

#app.run(host="0.0.0.0", port="8080")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
