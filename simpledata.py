from os import listdir
from numpy import asarray
from numpy import vstack
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import imageio
from numpy.random import randint
import numpy as np
import math
import random

def generate_real_samples(path, n_samples):
    files = listdir(path)

    input_files = [f for f in files if "input" in f]
    rand = randint(0, math.floor(len(files)/2)-n_samples)
    input_files = files[rand:rand+n_samples]
    output_files = [f.replace("input", "output") for f in input_files]

    X1, X2 = list(), list()
    for i in range(len(input_files)):
        pixels_in = imageio.imread(path + input_files[i])
        pixels_out = imageio.imread(path + output_files[i])
        
        pixels_in = image.img_to_array(pixels_in, dtype=np.uint16)
        pixels_out = image.img_to_array(pixels_out, dtype=np.uint16)

        slice_type = randint(0, 15)
        start1 = 0
        end1 = 0
        start2 = 0
        end2 = 0

        if slice_type == 0:
            end1 = 256
            end2 = 256
        elif slice_type == 1:
            start1 = 256
            end1 = 512
            end2 = 256
        elif slice_type == 2:
            start1 = 512
            end1 = 768
            end2 = 256
        elif slice_type == 3:
            start1 = 768
            end1 = 1024
            end2 = 256
        elif slice_type == 4:
            end1 = 256
            start2 = 256
            end2 = 512
        elif slice_type == 5:
            start1 = 256
            end1 = 512
            start2 = 256
            end2 = 512
        elif slice_type == 6:
            start1 = 512
            end1 = 768
            start2 = 256
            end2 = 512
        elif slice_type == 7:
            start1 = 768
            end1 = 1024
            start2 = 256
            end2 = 512
        elif slice_type == 8:
            end1 = 256
            start2 = 512
            end2 = 768
        elif slice_type == 9:
            start1 = 256
            end1 = 512
            start2 = 512
            end2 = 768
        elif slice_type == 10:
            start1 = 512
            end1 = 768
            start2 = 512
            end2 = 768
        elif slice_type == 11:
            start1 = 768
            end1 = 1024
            start2 = 512
            end2 = 768
        elif slice_type == 12:
            end1 = 256
            start2 = 768
            end2 = 1024
        elif slice_type == 13:
            start1 = 256
            end1 = 512
            start2 = 768
            end2 = 1024
        elif slice_type == 14:
            start1 = 512
            end1 = 768
            start2 = 768
            end2 = 1024
        elif slice_type == 15:
            start1 = 768
            end1 = 1024
            start2 = 768
            end2 = 1024

        pixels_in = pixels_in[start1:end1, start2:end2]
        pixels_out = pixels_out[start1:end1, start2:end2]

        rot_number = randint(0, 3)
        pixels_in = np.rot90(pixels_in, rot_number)
        pixels_in = np.rot90(pixels_in, rot_number)
        X1.append(pixels_in)
        X2.append(pixels_out)

    return np.array(X1), np.array(X2)




path = "/mnt/ml/SimpleData/"
XReal_A, XReal_B = generate_real_samples(path, 3)
print(XReal_A.shape)
print(XReal_B.shape)
