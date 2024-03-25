from numpy import load
from tensorflow.keras.preprocessing import image
import imageio
from numpy import zeros
from numpy import ones
from numpy.random import randint
import random
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
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
# import neptune

# define the discriminator model
def define_discriminator(gen_out_shape, tar_image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=gen_out_shape)
    # target image input
    in_target_image = Input(shape=tar_image_shape)
    # concatenate images channel-wise
    merged = Concatenate(axis=-1)([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
        merged
    )
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(0.3)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(0.2)(d)
    # C512
    d = Conv2D(512, (3, 3), strides=(1, 1), padding="valid", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(0.2)(d)

    # patch output
    d = Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init)(d)
    patch_out = Activation("sigmoid")(d)
    # define model
    model = Model([in_target_image, in_src_image], patch_out)
    # compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    # Flatten y_true and y_pred if necessary

    # Calculate binary cross-entropy loss
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    # model.compile(loss=reduce_mean_discriminator, optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)

    g = Conv2D(
        n_filters, (3, 3), padding="same", kernel_initializer=init
    )(g)
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
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

    g = Conv2D(
        n_filters, (3, 3), padding="same", kernel_initializer=init
    )(g)
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)

    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    g = Conv2D(32, (3, 3), padding="same", kernel_initializer=init)(in_image)
    skip1 = LeakyReLU(alpha=0.2)(g)

    # encoder model
    e1 = define_encoder_block(skip1, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 64)
    e3 = define_encoder_block(e2, 128)
    e4 = define_encoder_block(e3, 256)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)

    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(e7)
    b = Activation("relu")(b)

    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 256, dropout=False)
    d5 = decoder_block(d4, e3, 128, dropout=False)
    d6 = decoder_block(d5, e2, 64, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # output
    g = Conv2DTranspose(
        32, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(d7)
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    g = Concatenate()([g, skip1])

    g = Conv2D(
        32, (3, 3), padding="same", kernel_initializer=init
    )(g)
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)

    g = Conv2D(
        1, (3, 3), padding="same", kernel_initializer=init
    )(g)
    out_image = Activation("tanh")(g)
    # define model
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
    model.compile(
        loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100]
    )
    return model



def generate_real_samples(path, n_samples, patch_shape):
    files = listdir(path)

    
    input_files = [f for f in files if "Input" in f]
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

    y = ones((n_samples, patch_shape, patch_shape, 1))

    return np.array(X1), np.array(X2), y



# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
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
    pyplot.close()
    # save the generator model
    filename2 = "model_%06d.h5" % (step + 1)
    g_model.save(filename2)
    print(">Saved: %s and %s" % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, n_epochs=200, n_batch=4):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    print("n_patch: "  + str(n_patch))
    # unpack dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(4000/ n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, X_realB, y_real = generate_real_samples(dataPath, n_batch, n_patch)
        # print('xreal a shape: ' + str(X_realA.shape))
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print(">%d, d1[%.3f] d2[%.3f] g[%.3f]" % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (bat_per_epo) == 0:
            summarize_performance(i, g_model)


physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
dataPath = "/mnt/ml/SimpleData/"
# load image data
d_model = define_discriminator((256, 256, 1), (256, 256, 1))
g_model = define_generator((256, 256, 1))
# define the composite model
gan_model = define_gan(g_model, d_model, (256, 256, 1))
# train model
train(d_model, g_model, gan_model)
