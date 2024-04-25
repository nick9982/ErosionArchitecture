import numpy as np
from tensorflow.keras.preprocessing import image
import imageio
import cv2
from numpy import zeros
from numpy import ones
from numpy.random import randint
from os import listdir
import os

# generate real samples from our dataset
def generate_real_samples(path, n_samples, patch_shape, rangeBeg=0, rangeEnd=2199):
    # list all the files in the directory located at path
    files = listdir(path)
    
    # extract input data from the list
    input_files = [f for f in files if "Input" in f]
    # only take the input data within this range. Can differentiate between training, test, and validation data this way
    input_files = input_files[rangeBeg:rangeEnd]
    # generate a list of random numbers inside of the specified range then subtract 1 from each element in the list
    rand = randint(1, len(input_files)-n_samples)-1
    # gather the names of the input files as strings
    input_files = input_files[rand:rand+n_samples]
    # gather the names of the output files as strings
    output_files = [f.replace("Input", "Output") for f in input_files]

    # make sure there is a valid matching output file in the dataset
    output_file_valid = True
    for i in range(len(output_files)):
        if(not output_files[i] in files):
            output_file_valid = False
            break

    # if there isn't keep on generating file names until you get a valid set of input and output pairs
    while not output_file_valid:
        rand = randint(1, math.floor(len(input_files)/2)-n_samples)-1
        input_files = files[rand:rand+n_samples]
        output_files = [f.replace("Input", "Output") for f in input_files]

        output_file_valid = True
        for i in range(len(output_files)):
            if(not output_files[i] in files):
                output_file_valid = False
                break




    # define the lists which will store the images
    X1, X2 = list(), list()
    for i in range(len(input_files)):
        # get input image data
        pixels_in = imageio.imread(path + input_files[i])
        # get output image data
        pixels_out = imageio.imread(path + output_files[i])
        
        # turn the images into 16-bit numpy arrays
        pixels_in = image.img_to_array(pixels_in, dtype=np.uint16)
        pixels_out = image.img_to_array(pixels_out, dtype=np.uint16)

        # resize the images using Lanczos interpolation
        pixels_in = cv2.resize(pixels_in, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        pixels_out = cv2.resize(pixels_out, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # rotate the images a random number of times between 0 and 3
        rot_number = randint(0, 3)
        pixels_in = np.rot90(pixels_in, rot_number)
        pixels_out = np.rot90(pixels_out, rot_number)

        # normalize the values between -1 and 1
        pixels_in = (pixels_in - 32767.5) / 32767.5
        pixels_out = (pixels_out - 32767.5) / 32767.5

        # append the values to the X1 and X2 array
        X1.append(pixels_in)
        X2.append(pixels_out)

    # prepare the generated label to feed the discriminator
    y = zeros((n_samples, patch_shape, patch_shape, 1))

    return np.array(X1), np.array(X2), y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = ones((len(X), patch_shape, patch_shape, 1))
    return X, y

def fill_directories(n, start, end, real_path, fake_path, data_path):
    while n > 0:
        X_realA, X_realB, _ = generate_real_samples(data_path, 4, 30, start, end)
        X_fakeB, _ = generate_fake_samples(g_model, X_realA, 30)
        for i in range(len(X_realA)):
            real = np.array(X_realB[i] * 127.5 + 255, dtype=np.uint8)
            fake = np.squeeze(np.array(X_fakeB[i] * 127.5 + 255, dtype=np.uint8))
            real = np.expand_dims(real, axis=-1)
            fake = np.expand_dims(fake, axis=-1)
            real = np.repeat(real, 3, axis=-1)
            fake = np.repeat(fake, 3, axis=-1)
            cv2.imwrite(real_path + "real"+str(n-i)+".png", real)
            cv2.imwrite(fake_path + "fake"+str(n-i)+".png", fake)
        n = n - 4


def storeData(filename, data):
    f = open(filename, 'a')
    f.write(' ' + str(data))

def read_file_into_list(filename):
    lst = []
    with open(filename, 'r') as file:
        file_content = file.read()
        lst = [x for x in file_content.split(' ') if x != '']
        lst = np.array(lst, dtype=np.float32)
    return lst
