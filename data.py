# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from numpy import savez_compressed
import imageio
 
# load all images in a directory into memory
def load_images(path, size=(256,256)):
 src_list, tar_list = list(), list()
 # enumerate filenames in directory, assume all are images
 limit, idx = 2000, 0
 files = listdir(path)
 for filename in files:
 # load and resize the imageprintf
     if filename.split('_')[0] == 'Input':
         print(idx)
         output = filename.replace('Input', 'Output')
         if output not in files:
             continue

         pixels_in = imageio.imread(path + filename)
         pixels_out = imageio.imread(path + output)

         # convert to numpy array
         pixels_in = image.img_to_array(pixels_in)
         pixels_out = image.img_to_array(pixels_out)

         maxval = np.max(pixels_in)
         print(maxval)
         # split into satellite and map
         src_list.append(pixels_in)
         tar_list.append(pixels_out)
         if idx > limit:
             break
         idx += 1
 return [asarray(src_list), asarray(tar_list)]
 
# dataset path
path = '/mnt/ml/Data/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'erosion_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
