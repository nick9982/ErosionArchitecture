from numpy import int16, load
from tensorflow.keras.preprocessing import image
import imageio
import cv2
from numpy import zeros
from numpy import ones
from numpy.random import randint
import random
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.layers import Lambda
import tensorflow as tf
import numpy as np
from keras import backend as K
import math
from os import listdir
import os
from keras import backend
from matplotlib.animation import FuncAnimation
import subprocess
import re
from cleanfida.fid import compute_fid
import torch
# from tensorflow import autograd
# import neptune

fid_scores = []
iterations = []
iterations_loss = []
dL1_ls = []
dL2_ls = []
gen_loss = []
real_images = np.empty([0, 3, 256, 256])
fake_images = np.empty([0, 3, 256, 256])
def saveFIDScoresInGraph():
    global fid_scores
    global iterations
    pyplot.cla()  # Clear the current plot
    pyplot.axis("on")

    # Plot FID scores against iteration numbers
    pyplot.figure()
    pyplot.plot(iterations, fid_scores, marker='o', color='b')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('FID')
    pyplot.title('FID Scores')

    pyplot.savefig('fid_scores.png')


def graph_mse_be_loss():
    global dL1_ls
    global dL2_ls
    global gen_loss
    global iterations_loss
    pyplot.cla() # Clear the current plot
    pyplot.axis("on")

    pyplot.figure()
    pyplot.plot(iterations_loss, gen_loss, label='Gen Loss', color='b')
    pyplot.plot(iterations_loss, dL1_ls, label='DL1 Loss', color='r')
    pyplot.plot(iterations_loss, dL2_ls, label='DL2 Loss', color='g')

    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss')
    pyplot.title('Loss')

    pyplot.savefig('loss.png')

def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

file_idx = 0
def cache_img_for_fid(real_features, generated_features):
    global real_images
    global fake_images
    real = np.array(real_features* 32767.5 + 32767.5, dtype=np.uint16)
    fake = np.array(generated_features* 32767.5 + 32767.5, dtype=np.uint16)
    real = np.transpose(real, (0, 3, 1, 2))
    fake = np.transpose(fake, (0, 3, 1, 2))
    real = np.repeat(real, 3, axis=1)
    fake = np.repeat(fake, 3, axis=1)
    real_images = np.concatenate((real_images, real), axis=0)
    fake_images = np.concatenate((fake_images, fake), axis=0)

    

def save_imgs_for_fid():
    global real_images
    global fake_images
    global file_idx
    for i in range(len(real_images)):
        real = np.array(np.transpose(real_images[i], (1, 2, 0))/65535*255, dtype=np.int8)
        fake = np.array(np.transpose(fake_images[i], (1, 2, 0))/65535*255, dtype=np.int8)
        cv2.imwrite("tmp_fid_dir/real/real"+str(file_idx)+".png", real)
        cv2.imwrite("tmp_fid_dir/fake/fake"+str(file_idx)+".png", fake)
        file_idx += 1
        if file_idx == 3000:
            file_idx = 0

    real_images = np.empty([0, 3, 256, 256])
    fake_images = np.empty([0, 3, 256, 256])

def establish_initial_real_images_FID():
    cntr = 0
    files = listdir("tmp_fid_dir/real/")
    if len(files) > 2999: return;
    for i in range(100):
        _, X, _ = generate_real_samples("/home/nick/Projects/SimpleData/", 30, 30)
        for j in  range(len(X)):
            real = np.array(X[j]*255+127.5, dtype=np.int8)
            cv2.imwrite("tmp_fid_dir/real/real"+str(cntr)+".png", real)
            cntr += 1


def calculate_fid():
    global file_idx
    global real_images
    global fake_images

    real_images = np.empty([0, 3, 256, 256])
    fake_images = np.empty([0, 3, 256, 256])
    return compute_fid("tmp_fid_dir/real/", "tmp_fid_dir/fake/", mode="clean")

# clip model weights to a given hypercube
class ClipConstraint:
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
     # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}
 
const = ClipConstraint(0.01)
# define the discriminator model
def define_discriminator(gen_out_shape, tar_image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=123)
    # source image input
    in_src_image = Input(shape=gen_out_shape)
    # target image input
    in_target_image = Input(shape=tar_image_shape)
    # concatenate images channel-wise
    merged = Concatenate(axis=-1)([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_target_image, in_src_image], patch_out)
    # compile model
    # with this loss the discriminator beats  the generator in first epoch
    # opt = RMSprop(learning_rate=0.00005)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # Flatten y_true and y_pred if necessary

    # Calculate binary cross-entropy loss
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    # model.compile(loss=reduce_mean_discriminator, optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=145)
    # add downsampling layer
    g = Conv2D(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=1234)
    # add upsampling layer
    g = Conv2DTranspose(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.15)(g, training=True)
    # merge with skip connection
    g = LeakyReLU(alpha=0.2)(g)
    g = Concatenate()([g, skip_in])
    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=178)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    g = Conv2D(
        128, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(e2)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    e2 = LeakyReLU(alpha=0.2)(g)

    e3 = define_encoder_block(e2, 256)
    g = Conv2D(
        256, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(e3)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    e3 = LeakyReLU(alpha=0.2)(g)
    e4 = define_encoder_block(e3, 512)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(e4)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    e4 = LeakyReLU(alpha=0.2)(g)
    e5 = define_encoder_block(e4, 512)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(e5)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    e5 = LeakyReLU(alpha=0.2)(g)
    e6 = define_encoder_block(e5, 512)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(e6)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    e6 = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(e6)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    e6 = LeakyReLU(alpha=0.2)(g)
    e7 = define_encoder_block(e6, 512)

     # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)

    # decoder model
    d1 = decoder_block(b, e7, 512)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(d1)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    d1 = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(d1)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    d1 = LeakyReLU(alpha=0.2)(g)
    d2 = decoder_block(d1, e6, 512)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(d2)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    d2 = LeakyReLU(alpha=0.2)(g)
    d3 = decoder_block(d2, e5, 512)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(d3)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    d3 = LeakyReLU(alpha=0.2)(g)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    g = Conv2D(
        512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(d4)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    d4 = LeakyReLU(alpha=0.2)(g)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    g = Conv2D(
        256, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init
    )(d5)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    d5 = LeakyReLU(alpha=0.2)(g)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # opt = RMSprop(learning_rate=0.00005)
    model.compile(
        loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100]
    )
    return model



def generate_real_samples(path, n_samples, patch_shape, rangeBeg=0, rangeEnd=950):
    files = listdir(path)

    
    input_files = [f for f in files if "Input" in f]
    input_files = input_files[rangeBeg:rangeEnd]
    rand = randint(1, math.floor(len(input_files)/2)-n_samples)-1
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
        pixels_out = np.rot90(pixels_out, rot_number)
        pixels_in = (pixels_in - 32767.5) / 32767.5
        pixels_out = (pixels_out - 32767.5) / 32767.5
        X1.append(pixels_in)
        X2.append(pixels_out)

    y = zeros((n_samples, patch_shape, patch_shape, 1))

    return np.array(X1), np.array(X2), y



# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = ones((len(X), patch_shape, patch_shape, 1))
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, n_samples=3):
    # select a sample of input images
    X_realA, X_realB,  y_real= generate_real_samples(dataPath, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    pyplot.cla()
    pyplot.axis("off")
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(X_realA[i, :, :, 0], cmap="gray")
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis("off")
        pyplot.imshow(X_fakeB[i, :, :, 0], cmap="gray")
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis("off")
        pyplot.imshow(X_realB[i, :, :, 0], cmap="gray")
    # save plot to file
    filename1 = "plot_%06d.png" % (step + 1)
    pyplot.savefig(filename1)
    pyplot.savefig("current_plot.png")
    pyplot.close()
    # save the generator model
    filename2 = "model_%06d.h5" % (step + 1)
    g_model.save(filename2)
    print(">Saved: %s and %s" % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, n_epochs=200, n_batch=4, n_critic=1):
    global iterations
    global fid_scores
    global iterations_loss
    global dL1_ls
    global dL2_ls
    global gen_loss
    areThere2048Images = False
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # Metric to monitor
     #                         factor=0.5,         # Factor by which the learning rate will be reduced (new_lr = lr * factor)
     #                         patience=5,         # Number of epochs with no improvement after which learning rate will be reduced
     #                         min_learningrate=0.000001)        # Lower bound on the learning rate
    # unpack dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(4000/ n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        sum_dloss1 = 0
        sum_dloss2 = 0
        X_realA, X_realB, y_real = generate_real_samples(dataPath, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        for j in range(n_critic):
            if j != 0:
                X_realA, X_realB, y_real = generate_real_samples(dataPath, n_batch, n_patch)
                X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # print('xreal a shape: ' + str(X_realA.shape))
            # generate a batch of fake samples
            # update discriminator for real samples
            sum_dloss1 += d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            sum_dloss2 += d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, be_l, mse_l = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print(">%d, d1[%.3f] d2[%.3f] g[%.3f]" % (i + 1, sum_dloss1/n_critic, sum_dloss2/n_critic, g_loss))
        # summarize model performance
        cache_img_for_fid(X_realB, X_fakeB)
        if (i + 1) % 100 == 0:
            dL1_ls.append(sum_dloss1/n_critic)
            dL2_ls.append(sum_dloss2/n_critic)
            gen_loss.append(be_l)
            iterations_loss.append(i + 1)
            graph_mse_be_loss()
        if (i + 1) % 200 == 0:
            save_imgs_for_fid()
        if (i + 1) % (bat_per_epo) == 0:
            print('----------------------')
            fid_score = calculate_fid()
            print('FID: ' + str(fid_score))
            print('----------------------')
            fid_scores.append(fid_score)
            iterations.append(i+1)
            saveFIDScoresInGraph()
            summarize_performance(i, g_model)
            #reduce_lr.on_epoch_end(i+1/bat_per_epo, logs={'val_loss': fid_score})
        #if (i + 1) % 5000 == 0:
         #   x = 0


physical_devices = tf.config.experimental.list_physical_devices("GPU")

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
dataPath = "/home/nick/Projects/SimpleData/"
# load image data
d_model = define_discriminator((256, 256, 1), (256, 256, 1))
g_model = define_generator((256, 256, 1))
# define the composite model
gan_model = define_gan(g_model, d_model, (256, 256, 1))
# train model
establish_initial_real_images_FID()
train(d_model, g_model, gan_model)
