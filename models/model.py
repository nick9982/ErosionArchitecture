import time
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, BatchNormalization, LeakyReLU

# define the discriminator model
def define_discriminator(gen_out_shape, tar_image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=int(time.time()))
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
    # Adam optimizer
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # Compile model with binary crossentropy loss
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True, n_extra_layers=0):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=int(time.time())+1)
    # add downsampling layer
    g = Conv2D(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)

    # add dilated kernel n times
    for i in range(n_extra_layers):
        # add dilated convolutional layer with 5x5 kernel size
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
    init = RandomNormal(stddev=0.02, seed=int(time.time())+2)
    # add upsampling layer
    g = Conv2DTranspose(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # LeakyReLU activation function
    g = LeakyReLU(alpha=0.2)(g)
    # concatenate with skip connection
    g = Concatenate()([g, skip_in])

    # add a dilated kernel n times
    for i in range(n_extra_layers):
        # add dilated convolutional layer with 5x5 kernel size
        g = Conv2D(
            n_filters, (5, 5), strides=(1, 1), dilation_rate=(2, 2) , padding="same", kernel_initializer=init
        )(g)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        # LeakyReLU activation function
        g = LeakyReLU(alpha=0.2)(g)

    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=int(time.time())+3)
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
    d4 = decoder_block(d3, e4, 512, n_extra_layers=1)
    d5 = decoder_block(d4, e3, 256, n_extra_layers=1)
    d6 = decoder_block(d5, e2, 128)
    d7 = decoder_block(d6, e1, 64)
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
    # declare optimizer Adam 
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # compile the model with binary_ce for discriminator loss and mae for generator loss
    model.compile(
        loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100]
    )
    return model
