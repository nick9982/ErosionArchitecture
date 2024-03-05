import sys
sys.path.append('data')
sys.path.append('models')
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv3D
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout, Activation
from keras.models import Model
from erosionData import ErosionData
from matplotlib import pyplot
import numpy as np

def define_discriminator(imageShape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=imageShape)

    #C128
    d = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_image)
    d = MaxPooling2D(pool_size=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)

    #C256
    d = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d)
    d = MaxPooling2D(pool_size=(2, 2))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    #C512
    d = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d)
    d = MaxPooling2D(pool_size=(2, 2))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    #C512
    d = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d)
    d = MaxPooling2D(pool_size=(2, 2))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    #Patch output
    d = Conv2D(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = Activation('sigmoid')(d)

    model = Model(in_image, d)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


def encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)

    skip = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(layer_in)
    g = MaxPooling2D(pool_size=(2, 2))(skip)
    if batchnorm:
        g = BatchNormalization()(g, training=True)

    g = LeakyReLU(alpha=0.2)(g)
    return g, skip

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)

    #concatenate
    g = Concatenate()([layer_in, skip_in])
    g = Activation('relu')(g)

    #convolution, reduced features
    g = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)

    #upsampling
    g = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    
    return g

def define_generator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=image_shape)

    # encoder model
    e, s1 = encoder_block(in_image, 64, batchnorm=False) # o 128
    e, s2 = encoder_block(e, 128) # O 64
    e, s3 = encoder_block(e, 256) # O 32
    e, s4 = encoder_block(e, 512) # O 16
    e, s5 = encoder_block(e, 512) # O 8
    e, s6 = encoder_block(e, 512) # O 4
    e, s7 = encoder_block(e, 512) # O 2

    # bottom of U
    g = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(e)
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)

    g = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization()(g, training=True)
    g = Dropout(0.5)(g, training=True)

    # decoder model
    d = decoder_block(g, s7, 512) # O 4
    d = decoder_block(d, s6, 512) # O 8
    d = decoder_block(d, s5, 512) # O 16
    d = decoder_block(d, s4, 512) # O 32
    d = decoder_block(d, s3, 256, dropout=False) # O 64
    d = decoder_block(d, s2, 128, dropout=False) # O 128

    #output section
    d = Concatenate()([d, s1])
    d = Activation('relu')(d)
    
    d = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d, training=True)
    d = LeakyReLU(alpha=0.2)(d)

    out_image = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(d)
    out_image = Activation('tanh')(out_image)

    model = Model(in_image, out_image)
    return model

def define_gan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    in_src = Input(shape=image_shape)

    gen_out = g_model(in_src)
    dis_out = d_model(in_src)

    model = Model(in_src, [dis_out, gen_out])
    model.summary()

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model

def summarizePerformance(step, g_model, test_data_size, train_data_size, data, n_samples=3):
    data.setIndex(train_data_size)
    # plot real input images
    for i in range(n_samples):
        data.loadImage(i)
        inputImage, _ = data.getImages()
        pyplot.subplot(3, n_samples, 1+i)
        pyplot.axis('off')
        pyplot.imshow(inputImage[:, :, 0])
        data.iterateImage()
    
    data.setIndex(train_data_size)
    #plot generator output
    for i in range(n_samples):
        data.loadImage(i)
        inputImage, _ = data.getImages()
        inputImage = np.expand_dims(inputImage, axis=0)
        fake = g_model.predict(inputImage)
        fake = np.squeeze(fake)
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(fake)
        data.iterateImage()

    data.setIndex(train_data_size)

    for i in range(n_samples):
        data.loadImage(i)
        _, outputImage = data.getImages()
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(outputImage[:, :, 0])
        data.iterateImage()

    filename1 = 'training_data/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()

    filename2 = 'training_data/model_%06d.h5' % (step+1)

    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))



def train(d_model, g_model, gan_model, n_epochs=25, n_batch=1):
    train_data_size = 1000
    test_data_size = 100
    n_patch = d_model.output_shape[1]

    data = ErosionData('/mnt/ml/Data/')

    bat_per_epo = int(train_data_size / n_batch)

    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        if not data.loadImage(0):
            print('The data pointer exceeds the size of the dataset.')
            return

        inputImage, outputImage = data.getImages()
        # DEM = inputImage[:, :, 0]
        softness = inputImage[:, :, 1]
        strength = inputImage[:, :, 2]
        outputDEM = outputImage[:, :, 0]

        y_fake = np.zeros((1, 8, 8, 1))
        y_real = np.ones((1, 8, 8, 1))

        x_real = np.stack([outputDEM, softness, strength], axis=-1)
        x_real = np.expand_dims(x_real, axis=0)

        d_loss1 = d_model.train_on_batch(x_real, y_real)

        inputImage = np.expand_dims(inputImage, axis=0)
        fake = g_model.predict(inputImage)
        fake = np.squeeze(fake)
        x_fake = np.stack([fake, softness, strength], axis=-1)
        x_fake = np.expand_dims(x_fake, axis=0)

        d_loss2 = d_model.train_on_batch(x_fake, y_fake)

        g_loss, d_loss3, gen_loss = gan_model.train_on_batch(inputImage, [y_real, x_real])

        
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        print('d3[%.3f], gen[%.3f]' % (d_loss3, gen_loss))

        if (i+1) % (train_data_size) == 0:
            summarizePerformance(i, g_model, test_data_size, train_data_size, data)
            data.setIndex(0)
