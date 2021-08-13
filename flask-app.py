from flask import Flask, request, jsonify
import werkzeug

import os
import time
import tensorflow_hub as hub
import tensorflow as tf
from functions import preprocess_image, save_image, plot_image, downscale_image
#import matplotlib.pyplot as plt

app = Flask(__name__)
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


def esrgan(image):
    SAVED_MODEL_PATH = "./model"
    IMAGE_PATH = image
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


def superResolution(image):
# Declaring Constants
    IMAGE_PATH = image
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
    plot_image(tf.squeeze(fake_image), title="UpscalSuper_Resolution")
    save_image(tf.squeeze(fake_image), filename="UpscalSuper_Resolution")



    # plt.rcParams['figure.figsize'] = [15, 10]
    # fig, axes = plt.subplots(1, 3)
    # fig.tight_layout()
    # plt.subplot(131)
    # plot_image(tf.squeeze(hr_image), title="Original")
    # plt.subplot(132)
    # fig.tight_layout()
    # plot_image(tf.squeeze(lr_image), "x4 Bicubic")
    # plt.subplot(133)
    # fig.tight_layout()
    # plot_image(tf.squeeze(fake_image), "Super Resolution")
    # plt.savefig("ESRGAN_DIV2K.jpg", bbox_inches="tight")
    # print("PSNR: %f" % psnr)

    # Plotting Super Resolution Image


@app.route('/upload', methods=['POST'])
def updload():
    if(request.method == 'POST'):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename("input.jpg")
        imagefile.save("./uploadimages/" + filename)

        image = "./uploadimages/input.jpg"
        esrgan(image)
        superResolution(image)
        return jsonify({
            "message": "image uploaded succesfuly"

        })
        
@app.route('/download/<fname>', methods=['GET'])
def download(fname):
    return send_file(fname, as_attachment=True)

#app.run(host="0.0.0.0", port="8080")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")


# IMAGE_PATH = "original.png"
# SAVED_MODEL_PATH = "./model"

# hr_image = preprocess_image(IMAGE_PATH)

# # Plotting Original Resolution image
# plot_image(tf.squeeze(hr_image), title="Original Image")
# save_image(tf.squeeze(hr_image), filename="Original Image")

# model = hub.load(SAVED_MODEL_PATH)

# start = time.time()
# fake_image = model(hr_image)
# fake_image = tf.squeeze(fake_image)
# print("Time Taken: %f" % (time.time() - start))

# # Plotting Super Resolution Image
# plot_image(tf.squeeze(fake_image), title="Super Resolution")
# save_image(tf.squeeze(fake_image), filename="Super Resolution")
