import keras
import tensorflow as tf
from os import listdir
from numpy.random import randint
import imageio
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import time

loaded_model = tf.keras.models.load_model("model_200000.h5")

def generate_real_samples(path, n_samples, patch_shape, rangeBeg=0, rangeEnd=2199):
    files = listdir(path)

    
    input_files = [f for f in files if "Input" in f]
    input_files = input_files[rangeBeg:rangeEnd]
    rand = randint(1, len(input_files)-n_samples)-1
    input_files = input_files[rand:rand+n_samples]
    output_files = [f.replace("Input", "Output") for f in input_files]

    output_file_valid = True
    for i in range(len(output_files)):
        if(not output_files[i] in files):
            output_file_valid = False
            break

    while not output_file_valid:
        rand = randint(1, math.floor(len(input_files)/2)-n_samples)-1
        input_files = files[rand:rand+n_samples]
        output_files = [f.replace("Input", "Output") for f in input_files]

        output_file_valid = True
        for i in range(len(output_files)):
            if(not output_files[i] in files):
                output_file_valid = False
                break




    X1, X2 = list(), list()
    for i in range(len(input_files)):
        pixels_in = imageio.imread(path + input_files[i])
        pixels_out = imageio.imread(path + output_files[i])
        
        pixels_in = image.img_to_array(pixels_in, dtype=np.uint16)
        pixels_out = image.img_to_array(pixels_out, dtype=np.uint16)

        pixels_in = cv2.resize(pixels_in, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        pixels_out = cv2.resize(pixels_out, (256, 256), interpolation=cv2.INTER_LANCZOS4)


        rot_number = randint(0, 3)
        pixels_in = np.rot90(pixels_in, rot_number)
        pixels_out = np.rot90(pixels_out, rot_number)
        pixels_in = (pixels_in - 32767.5) / 32767.5
        pixels_out = (pixels_out - 32767.5) / 32767.5
        X1.append(pixels_in)
        X2.append(pixels_out)

    y = np.zeros((n_samples, patch_shape, patch_shape, 1))

    return np.array(X1), np.array(X2), y


X_realA, X_realB, Y_real = generate_real_samples('/home/dave01/ComplexData/', 32, 1, 2000, 2200)


def saveAllImages(inp, o, real):
    for i in range(len(o)):
        im = np.array(inp[i] * 32767.5 + 32767.5, dtype= np.uint16)
        cv2.imwrite("inp_" + str(i) + ".png", im)
        generated = np.array(o[i] * 32767.5 + 32767.5, dtype=np.uint16)
        cv2.imwrite("generated_" + str(i) +".png", generated)
        x = np.array(real[i, :, :] * 32767.5 + 32767.5, dtype=np.uint16)
        cv2.imwrite("real_"+str(i)+".png", x)


for i in range(30):
    start = time.time()
    o = loaded_model.predict(X_realA)
    print("Time: " + str(time.time() - start))

    saveAllImages(X_realA, o, X_realB)



def ErosionOnFile(filename, loaded_model, outname):
    pixels_in = imageio.imread(filename)
    
    pixels_in = image.img_to_array(pixels_in, dtype=np.uint8)

    pixels_in = (pixels_in - 127.5) / 127.5
    pixels_in = pixels_in[:, :, 0]
    pixels_in = np.expand_dims(pixels_in, axis=0)


    o = loaded_model.predict(pixels_in)
    o = np.array(o * 127.5 + 127.5, dtype = np.uint8)

    o = np.squeeze(o)
    cv2.imwrite(outname, o)

# ErosionOnFile("hllogo.png", loaded_model, "logo.png")
# ErosionOnFile("blurredhllogo.png", loaded_model, "logobl.png")
# ErosionOnFile("test.png", loaded_model, "o.png")

