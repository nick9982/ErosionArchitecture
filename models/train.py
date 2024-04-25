from data import generate_real_samples, generate_fake_samples, fill_directories, storeData
from model import define_discriminator, define_generator, define_gan
import numpy as np

# train pix2pix models
# dataset size I set to 4000 because of our data augmentation. The data set training size is truly 2200 images. 
# the augmentation makes this into 8800 images, but there will be repeats in random selection, so I chose each
# epoch to process 4000 images
def train(d_model, g_model, gan_model, dataPath, n_epochs=200, n_batch=4, n_critic=1, dataset_size=4000):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # calculate the number of batches in each epoch
    bat_per_epo = int(dataset_size / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        sum_dloss1 = 0
        sum_dloss2 = 0
        # load a batch of real samples
        X_realA, X_realB, y_real = generate_real_samples(dataPath, n_batch, n_patch)
        # load a batch of generated samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        # some CGAN architectures require multiple discriminator passes in order to train
        # ours does not but I still allow the user to specify if they would like to have multiple
        # critic passes
        for j in range(n_critic):
            # if not first pass, we can generate new samples
            if j != 0:
                X_realA, X_realB, y_real = generate_real_samples(dataPath, n_batch, n_patch)
                X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

            # run the discriminator on real image
            sum_dloss1 += d_model.train_on_batch([X_realA, X_realB], y_real)
            # run the discriminator on fake image
            sum_dloss2 += d_model.train_on_batch([X_realA, np.squeeze(X_fakeB)], y_fake)
        # update the generator
        g_loss, be_l, mae_l = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print(">%d, d1[%.3f] d2[%.3f] g[%.3f]" % (i + 1, sum_dloss1/n_critic, sum_dloss2/n_critic, g_loss))
        # every 200 iterations, 5 times per epoch we plot the loss of the discriminator
        if (i + 1) % 200 == 0:
            # dL1_ls is the failure to identify that a real image is real
            storeData('dL1_ls.txt', sum_dloss1/n_critic)
            # dL2_ls is the failure to identify that a fake image is fake 
            storeData('dL2_ls.txt', sum_dloss2/n_critic)
            # store the iteration value that the discriminator losses were collected at
            storeData('loss_iterations.txt', i+1)
            # save matplotlib graph of discriminator loss
            graphGANLoss()
            # twice per epoch we evaluate lpips and mae. We use these to create a graph
        if (i + 1) % 500 == 0:
            # fill_directories specified with validation data
            fill_directories(128, 2300, 2450, 'tmp_validation_data/real/', 'tmp_validation_data/fake/', dataPath)
            # fill_directories specified with training data
            fill_directories(128, 0, 2299, 'tmp_training_data/real/', 'tmp_training_data/fake/', dataPath)
            # evaluate lpips for both training data and validation data
            storeData('lpips_val.txt', lpips_eval('tmp_validation_data/real/', 'tmp_validation_data/fake/'))
            storeData('lpips_training.txt', lpips_eval('tmp_training_data/real/', 'tmp_training_data/fake/'))
            storeData('lpips_iterations.txt', i+1)
            # evaluate mae for both training and validation data
            storeData('mae.txt', mae('tmp_training_data/real/', 'tmp_training_data/fake/'))
            storeData('mae_val.txt', mae('tmp_validation_data/real/', 'tmp_validation_data/fake/'))
            # plot the LPIPS scores and the mae scores
            saveLPIPScores()
            plotMAE()
        if (i + 1) % bat_per_epo == 0:
            summarize_performance(i, g_model)

import subprocess
import os
cwd = os.getcwd()
command = f'rm {cwd}/*.txt'
subprocess.run(command, shell=True)

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # balance memory growth the same across all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

dataPath = "/home/dave01/ComplexData/"
# load d_model and g_model
d_model = define_discriminator((256, 256, 1), (256, 256, 1))
g_model = define_generator((256, 256, 1))
# define the composite model
gan_model = define_gan(g_model, d_model, (256, 256, 1))
# train model
train(d_model, g_model, gan_model, dataPath)
