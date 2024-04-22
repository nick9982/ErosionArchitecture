from numpy import int16, load
import time
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
import torch
from lpips.lpips import LPIPS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

key_path = '/home/dave01/erosionmodel-g-api-key.json'

SCOPES = ['https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=credentials)

lpips = LPIPS(net='alex')

real_images = np.empty([0, 3, 256, 256])
fake_images = np.empty([0, 3, 256, 256])

def save3ImagesToDrive():
    X_realA, X_realB, _ = generate_real_samples("/home/dave01/ComplexData/", 3, 30, 2200, 2450)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 30)
    for i in range(len(X_realA)):
        real = np.array(X_realB[i] * 32767.5 + 32767.5, dtype=np.uint16)
        fake = np.array(X_fakeB[i] * 32767.5 + 32767.5, dtype=np.uint16)
        inp = np.array(X_realA[i] * 32767.5 + 32767.5, dtype=np.uint16)
        real = np.expand_dims(real, axis=-1)
        inp = np.expand_dims(inp, axis=-1)
        cv2.imwrite("real_out_"+str(i)+".png", real)
        cv2.imwrite("fake_out_"+str(i)+".png", fake)
        cv2.imwrite("inp_"+str(i)+".png", inp)
        save_file_to_gdrive('real_out_'+str(i)+'.png', '/home/dave01/Project/ErosionArchitecture/models/real_out_'+str(i)+'.png')
        save_file_to_gdrive('fake_out_'+str(i)+'.png', '/home/dave01/Project/ErosionArchitecture/models/fake_out_'+str(i)+'.png')
        save_file_to_gdrive('inp_'+str(i)+'.png', '/home/dave01/Project/ErosionArchitecture/models/inp_'+str(i)+'.png')


def saveLPIPScores():

    lpips_scores = read_file_into_list('lpips.txt')
    lpips_iterations = read_file_into_list('lpips_iterations.txt')
    pyplot.cla()
    pyplot.axis("on")

    pyplot.figure()
    pyplot.plot(lpips_iterations, lpips_scores, label='lpips')

    pyplot.xlabel('Iteration')
    pyplot.ylabel('lpips')
    pyplot.title('lpips')

    pyplot.savefig('lpips.png')
    save_file_to_gdrive('lpips.png', '/home/dave01/Project/ErosionArchitecture/models/lpips.png')

def deleteAllFilesWithName(fname):
    hasRequestBeenSatisfied = False
    while not hasRequestBeenSatisfied:
        try:
            response = drive_service.files().list(
                    q="name='"+fname+"' and '1u9k17k1xTvZJOjS4-oIvIRB5ozuho8v0' in parents",
                    fields='files(id, name)'
                    ).execute()
            hasRequestBeenSatisfied = True
            for file in response.get('files', []):
                file_id = file.get('id')
                drive_service.files().delete(fileId=file_id).execute()
        except HttpError:
            print("google connection error")
            print("wait 1 minute to request again")
            time.sleep(60)

import requests

def save_file_to_gdrive(fname, fpath):

    #deleteAllFilesWithName(fname)
    time.sleep(0.4)
    hasRequestBeenSatisfied = False
    #while not hasRequestBeenSatisfied:
    #    try:
    #        file_metadata = {'name': fname, "parents": ['1u9k17k1xTvZJOjS4-oIvIRB5ozuho8v0']}
    #        media = MediaFileUpload(fpath, mimetype='image/png', resumable=True)
    #        f = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    #        hasRequestBeenSatisfied = True
    #    except HttpError:
    #        print("google connection error")
    #        print("Wait 1 minute to request again")
    #        time.sleep(60)

def graphGANLoss():
    dL1_ls = read_file_into_list('dL1_ls.txt')
    dL2_ls = read_file_into_list('dL2_ls.txt')
    gen_loss = read_file_into_list('gen_ls.txt')
    iterations_loss = read_file_into_list('loss_iterations.txt')

    pyplot.cla() 
    pyplot.axis("on")

    pyplot.figure()
    pyplot.plot(iterations_loss, gen_loss, label='Gen Loss', color='b')
    pyplot.plot(iterations_loss, dL1_ls, label='DL1 Loss', color='r')
    pyplot.plot(iterations_loss, dL2_ls, label='DL2 Loss', color='g')

    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss')
    pyplot.title('Loss')

    pyplot.savefig('loss.png')
    pyplot.close()
    save_file_to_gdrive('loss.png', '/home/dave01/Project/ErosionArchitecture/models/loss.png')

file_idx = 0

def fill_validation_directories(n):
    while n > 0:
        X_realA, X_realB, _ = generate_real_samples("/home/dave01/SimpleData/", 4, 30, 901, 1050)
        X_fakeB, _ = generate_fake_samples(g_model, X_realA, 30)
        for i in range(len(X_realA)):
            real = np.array(X_realB[i] * 127.5 + 255, dtype=np.uint8)
            fake = np.squeeze(np.array(X_fakeB[i] * 127.5 + 255, dtype=np.uint8))
            real = np.expand_dims(real, axis=-1)
            fake = np.expand_dims(fake, axis=-1)
            real = np.repeat(real, 3, axis=-1)
            fake = np.repeat(fake, 3, axis=-1)
            cv2.imwrite("tmp_validation_data/real/real"+str(n-i)+".png", real)
            cv2.imwrite("tmp_validation_data/fake/fake"+str(n-i)+".png", fake)
        n = n - 4

def storeMSEAndPSNR(mse):
    mse_file = open('mse.txt', 'a')
    psnr_file = open('psnr.txt', 'a')
    mse_file.write(' ' + str(mse))
    psnr_file.write(' ' + str(20 * math.log(65535, 10) - 10 * math.log(mse, 10)))

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

def plotMSEAndPSNR():
    iterations = read_file_into_list('loss_iterations.txt')
    mse = read_file_into_list('mse.txt')
    pyplot.cla()
    pyplot.figure()
    pyplot.plot(iterations, mse, color='b')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('mse')
    pyplot.title('MSE Scores')

    pyplot.savefig('mse.png')
    save_file_to_gdrive('mse.png', '/home/dave01/Project/ErosionArchitecture/models/mse.png')

    pyplot.close()
    psnr = read_file_into_list('psnr.txt')
    pyplot.cla()
    pyplot.figure()
    pyplot.plot(iterations, psnr, color ='b')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('psnr')
    pyplot.title('PSNR Scores')

    pyplot.savefig('psnr.png')
    save_file_to_gdrive('psnr.png', '/home/dave01/Project/ErosionArchitecture/models/psnr.png')

    pyplot.close()

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
def define_encoder_block(layer_in, n_filters, batchnorm=True, n_extra_layers=0):
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

    for i in range(n_extra_layers):
        g = Conv2D(
            n_filters, (5, 5), strides=(1, 1), dilation_rate=(2, 2), padding="same", kernel_initializer=init
        )(g)
        # conditionally add batch normalization
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)
        
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=False, n_extra_layers=0):
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

    for i in range(n_extra_layers):
        g = Conv2D(
            n_filters, (5, 5), strides=(1, 1), dilation_rate=(2, 2) , padding="same", kernel_initializer=init
        )(g)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        # conditionally add dropout
        if dropout:
            g = Dropout(0.15)(g, training=True)
        # merge with skip connection
        g = LeakyReLU(alpha=0.2)(g)

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
    e3 = define_encoder_block(e2, 256, n_extra_layers=1)
    e4 = define_encoder_block(e3, 512, n_extra_layers=1)
    e5 = define_encoder_block(e4, 512, n_extra_layers=1)
    e6 = define_encoder_block(e5, 512, n_extra_layers=2)
    e7 = define_encoder_block(e6, 512, n_extra_layers=2)

     # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)

    # decoder model
    d1 = decoder_block(b, e7, 512, n_extra_layers=2)
    d2 = decoder_block(d1, e6, 512, n_extra_layers=1)
    d3 = decoder_block(d2, e5, 512, n_extra_layers=1)
    d4 = decoder_block(d3, e4, 512, dropout=False, n_extra_layers=1)
    d5 = decoder_block(d4, e3, 256, dropout=False, n_extra_layers=1)
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
    X_realA = np.expand_dims(X_realA, axis=-1)
    X_realB = np.expand_dims(X_realB, axis=-1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    x = np.arange(0, X_realA.shape[2])
    y = np.arange(0, X_realA.shape[1])
    X, Y = np.meshgrid(x, y)
    pyplot.cla()
    pyplot.axis("off")
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis("off")
    ax.view_init(45, 215)
    ax.plot_surface(X, Y, X_realA[0, :, :, 0], cmap='inferno', alpha=0.8, linewidth=0, antialiased=False, rcount=200, ccount=200)
    fig.savefig("current_inp_plot.png")
    save_file_to_gdrive('current_inp_plot.png', '/home/dave01/Project/ErosionArchitecture/models/current_inp_plot.png')

    pyplot.cla()
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis("off")
    ax.view_init(45, 215)
    ax.plot_surface(X, Y, X_fakeB[0, :, :, 0], cmap='inferno', alpha=0.8, linewidth=0, antialiased=False, rcount=200, ccount=200)
    fig.savefig("current_fake_plot.png")
    save_file_to_gdrive('current_fake_plot.png', '/home/dave01/Project/ErosionArchitecture/models/current_fake_plot.png')

    pyplot.cla()
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis("off")
    ax.view_init(45, 215)
    ax.plot_surface(X, Y, X_realB[0, :, :, 0], cmap='inferno', alpha=0.8, linewidth=0, antialiased=False, rcount=200, ccount=200)
    fig.savefig("current_real_plot.png")
    save_file_to_gdrive('current_real_plot.png', '/home/dave01/Project/ErosionArchitecture/models/current_real_plot.png')


    pyplot.cla()
    pyplot.axis("off")
 #   plot real source images
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

    filename1 = "plot_%06d.png" % (step + 1)
    pyplot.savefig(filename1)
    pyplot.savefig("current_plot.png")
    save_file_to_gdrive('current_plot.png', '/home/dave01/Project/ErosionArchitecture/models/current_plot.png')

    pyplot.close()
    # save the generator model
    filename2 = "model_%06d.h5" % (step + 1)
    g_model.save(filename2)
    print(">Saved: %s and %s" % (filename1, filename2))

def lpips_eval():
    files = listdir("tmp_validation_data/real/")
    sum_lpip = 0
    pix_in = np.empty([0, 3, 256, 256])
    pix_out = np.empty([0, 3, 256, 256])
    batch = 64
    cnt = 1
    comps = 0
    for real in files:
        fake = 'tmp_validation_data/fake/' + real.replace('real', 'fake')
        pixels_in = imageio.imread('tmp_validation_data/real/'+real)
        pixels_out = imageio.imread(fake)
        
        pixels_in = image.img_to_array(pixels_in)
        pixels_out = image.img_to_array(pixels_out)

        pixels_in = np.expand_dims(pixels_in, axis=0)
        pixels_out = np.expand_dims(pixels_out, axis=0)
        pixels_in = np.transpose(pixels_in, (0, 3, 1, 2))
        pixels_out = np.transpose(pixels_out, (0, 3, 1, 2))
        pix_in = np.append(pix_in, pixels_in, axis=0)
        pix_out = np.append(pix_out, pixels_out, axis=0)

        if cnt % batch == 0:
            pix_in = torch.tensor(pix_in).float()
            pix_out = torch.tensor(pix_out).float()
            res = lpips.forward(pix_in, pix_out)
            sum_lpip = sum_lpip + torch.mean(res).item()
            pix_in = np.empty([0, 3, 256, 256])
            pix_out = np.empty([0, 3, 256, 256])
            comps += 1
        cnt += 1
    if comps == 0:
        return 0
    return sum_lpip/comps


# train pix2pix models
def train(d_model, g_model, gan_model, n_epochs=200, n_batch=4, n_critic=1):
    global fid_scores
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
            sum_dloss1 += d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            sum_dloss2 += d_model.train_on_batch([X_realA, np.squeeze(X_fakeB)], y_fake)
        # update the generator
        g_loss, be_l, mse_l = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print(">%d, d1[%.3f] d2[%.3f] g[%.3f]" % (i + 1, sum_dloss1/n_critic, sum_dloss2/n_critic, g_loss))
        # summarize model performance
        #cache_img_for_fid(X_realB, X_fakeB)
        if (i + 1) % 200 == 0:
            storeData('mse.txt', mse_l)
            storeData('psnr.txt', 20 * math.log(65535, 10) - 10 * math.log(mse_l, 10))
            storeData('dL1_ls.txt', sum_dloss1/n_critic)
            storeData('dL2_ls.txt', sum_dloss2/n_critic)
            storeData('gen_ls.txt', g_loss)
            storeData('loss_iterations.txt', i+1)
            graphGANLoss()
            plotMSEAndPSNR()
        if (i + 1) % 500 == 0:
            save3ImagesToDrive()
            fill_validation_directories(200)
            storeData('lpips.txt', lpips_eval())
            storeData('lpips_iterations.txt', i+1)
            saveLPIPScores()
        if (i + 1) % 1000 == 0:
            summarize_performance(i, g_model)
            #reduce_lr.on_epoch_end(i+1/bat_per_epo, logs={'val_loss': fid_score})
        #if (i + 1) % 5000 == 0:
         #   x = 0


#physical_devices = tf.config.experimental.list_physical_devices("GPU")

#if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import subprocess
cwd = os.getcwd()
command = f'rm {cwd}/*.txt'
subprocess.run(command, shell=True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

dataPath = "/home/dave01/ComplexData/"
# load image data
d_model = define_discriminator((256, 256, 1), (256, 256, 1))
g_model = define_generator((256, 256, 1))
# define the composite model
gan_model = define_gan(g_model, d_model, (256, 256, 1))
# train model
train(d_model, g_model, gan_model)