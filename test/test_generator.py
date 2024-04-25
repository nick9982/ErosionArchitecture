import keras
import tensorflow as tf
from os import listdir
from numpy.random import randint
import imageio
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import time
from models.data import generate_real_samples

def saveAllImages(inp, o, real):
    for i in range(len(o)):
        im = np.array(inp[i] * 32767.5 + 32767.5, dtype= np.uint16)
        cv2.imwrite("inp_" + str(i) + ".png", im)
        generated = np.array(o[i] * 32767.5 + 32767.5, dtype=np.uint16)
        cv2.imwrite("generated_" + str(i) +".png", generated)
        x = np.array(real[i, :, :] * 32767.5 + 32767.5, dtype=np.uint16)
        cv2.imwrite("real_"+str(i)+".png", x)

# use model on individual file that uses 8-bit images
def ErosionOnFileWith8_BitImages(filename, loaded_model, outname):
    # read the specified file
    pixels_in = imageio.imread(filename)
    
    # convert the image to a numpy array
    pixels_in = image.img_to_array(pixels_in, dtype=np.uint8)

    # downscale the image using lanczos interpolation
    pixels_in = cv2.resize(pixels_in, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # normalize the image data
    pixels_in = (pixels_in - 127.5) / 127.5
    pixels_in = pixels_in[:, :, 0]
    pixels_in = np.expand_dims(pixels_in, axis=0)

    # make prediction
    o = loaded_model.predict(pixels_in)

    # denormalize output data
    o = np.array(o * 127.5 + 127.5, dtype = np.uint8)

    # make data valid image format
    o = np.squeeze(o)

    # save output
    cv2.imwrite(outname, o)

# use model on individual file that uses 16-bit images
def ErosionOnFileWith16_BitImages(filename, loaded_model, outname):
    # read the specified file
    pixels_in = imageio.imread(filename)
    
    # convert the image to a numpy array
    pixels_in = image.img_to_array(pixels_in, dtype=np.uint8)

    # downscale the image using lanczos interpolation
    pixels_in = cv2.resize(pixels_in, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # normalize the image data
    pixels_in = (pixels_in - 32677.5) / 32677.5
    pixels_in = pixels_in[:, :, 0]
    pixels_in = np.expand_dims(pixels_in, axis=0)

    # make prediction
    o = loaded_model.predict(pixels_in)

    # denormalize output data
    o = np.array(o * 32677.5 + 32677.5, dtype = np.uint8)

    # make data valid image format
    o = np.squeeze(o)

    # save output
    cv2.imwrite(outname, o)

# load in the keras model from the h5 file
loaded_model = tf.keras.models.load_model("model_200000.h5")

# generate inputs
X_realA, X_realB, Y_real = generate_real_samples('/home/dave01/ComplexData/', 32, 1, 2200, 2450)

# make prediction
o = loaded_model.predict(X_realA)

# ErosionOnFileWith16_BitImages('custom_terrain.png', loaded_model, 'output_terrain.png')