from matplotlib import pyplot
from data import read_file_into_list
import numpy as np
from lpips.lpips import LPIPS
import torch

def graphGANLoss():
    # Read discriminator losses from stored text files
    dL1_ls = read_file_into_list('dL1_ls.txt')
    dL2_ls = read_file_into_list('dL2_ls.txt')
    iterations_loss = read_file_into_list('loss_iterations.txt')

    pyplot.cla() 
    pyplot.axis("on")

    # Plot the discriminator losses
    pyplot.figure()
    pyplot.plot(iterations_loss, dL1_ls, label='DL1 Loss', color='r')
    pyplot.plot(iterations_loss, dL2_ls, label='DL2 Loss', color='b')

    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss')
    pyplot.title('Loss')

    pyplot.legend(loc='upper right')

    # Save the discriminator losses to file loss.png
    pyplot.savefig('loss.png')
    pyplot.close()

def saveLPIPScores():
    # Read LPIPS score for validation and training datasets
    lpips_val_scores = read_file_into_list('lpips_val.txt')
    lpips_training_scores = read_file_into_list('lpips_training.txt')
    lpips_iterations = read_file_into_list('lpips_iterations.txt')
    pyplot.cla()
    pyplot.axis("on")

    # Plot the LPIPS scores for validation and training
    pyplot.figure()
    pyplot.plot(lpips_iterations, lpips_val_scores, label='val', color='b')
    pyplot.plot(lpips_iterations, lpips_training_scores, label='training', color='r')

    pyplot.xlabel('Iteration')
    pyplot.ylabel('lpips')
    pyplot.title('lpips')

    pyplot.legend(loc='upper right')

    # Save the plot in lpips.png
    pyplot.savefig('lpips.png')

def plotMAE():
    # Read MAE score for validation and training datasets
    iterations = read_file_into_list('lpips_iterations.txt')
    mae = read_file_into_list('mae.txt')
    mae_val = read_file_into_list('mae_val.txt')
    pyplot.cla()
    pyplot.figure()

    # Plot the mae score for validation and training
    pyplot.plot(iterations, mae_val, label='val', color='b')
    pyplot.plot(iterations, mae, label='training', color='r')
    pyplot.legend(loc='upper right')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('mae')
    pyplot.title('MAE Scores')

    # save the mae score to mae.png file
    pyplot.savefig('mae.png')
    pyplot.close()

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

    # declare a meshgrid to plot the 3d models of our data
    x = np.arange(0, X_realA.shape[2])
    y = np.arange(0, X_realA.shape[1])
    X, Y = np.meshgrid(x, y)
    pyplot.cla()

    # The following 3 chunks of code are plotting 3d models of our DEM's. The input, generated, and real images respectively
    pyplot.axis("off")
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis("off")
    ax.view_init(45, 215)
    ax.plot_surface(X, Y, X_realA[0, :, :, 0], cmap='inferno', alpha=0.8, linewidth=0, antialiased=False, rcount=200, ccount=200)
    fig.savefig("current_inp_plot.png")

    pyplot.cla()
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis("off")
    ax.view_init(45, 215)
    ax.plot_surface(X, Y, X_fakeB[0, :, :, 0], cmap='inferno', alpha=0.8, linewidth=0, antialiased=False, rcount=200, ccount=200)
    fig.savefig("current_fake_plot.png")

    pyplot.cla()
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis("off")
    ax.view_init(45, 215)
    ax.plot_surface(X, Y, X_realB[0, :, :, 0], cmap='inferno', alpha=0.8, linewidth=0, antialiased=False, rcount=200, ccount=200)
    fig.savefig("current_real_plot.png")


    # This plot will create the 3x3 grid of input, generated, actual DEM's
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

    # save the plot
    filename1 = "plot_%06d.png" % (step + 1)
    pyplot.savefig(filename1)
    pyplot.savefig("current_plot.png")

    pyplot.close()
    # save the generator model
    filename2 = "model_%06d.h5" % (step + 1)
    g_model.save(filename2)

# this function calculates mae of two directories with pairs of images
def mae(real_dir, fake_dir):
    # list all of the files in the real directory
    files = listdir(real_dir)
    sm = 0
    cnt = 0 
    for real in files:
        # replace the real label with fake label to get the fake image filenames
        fake = fake_dir + real.replace('real', 'fake')

        # read the images located at the specified filepath
        pixels_in = imageio.imread(real_dir + real)
        pixels_out = imageio.imread(fake)

        # convert input images into numpy arrays
        pixels_in = image.img_to_array(pixels_in)
        pixels_out = image.img_to_array(pixels_out)
        
        # calculate mae
        mae_loss = float(tf.reduce_mean(tf.abs(pixels_in - pixels_out)).numpy())
        # add mae to the sum after iterating
        sm += mae_loss
        cnt += 1

    # find the average mae for the images in said directory
    return mae_loss/cnt

def lpips_eval(real_dir, fake_dir):
    # find all the files in the real image directory
    files = listdir(real_dir)
    sum_lpip = 0

    # declare an array with 3-channel rgb images
    pix_in = np.empty([0, 3, 256, 256])
    pix_out = np.empty([0, 3, 256, 256])
    # declares a number to use as the batchsize for the lpips eval
    batch = 64
    # cnt is used to locate the end of the batch in the loop so it knows when to run lpips and clear the array
    cnt = 1
    # comps is used to average the sum of lpip evals that were collected on the data
    comps = 0
    for real in files:
        # list all fake images
        fake = fake_dir + real.replace('real', 'fake')

        # read images from directory
        pixels_in = imageio.imread(real_dir + real)
        pixels_out = imageio.imread(fake)
        
        # convert the images into numpy arrays
        pixels_in = image.img_to_array(pixels_in)
        pixels_out = image.img_to_array(pixels_out)

        # expand on the 0th axis to make image rgb
        pixels_in = np.expand_dims(pixels_in, axis=0)
        pixels_out = np.expand_dims(pixels_out, axis=0)

        # re-order the image's numpy representation
        pixels_in = np.transpose(pixels_in, (0, 3, 1, 2))
        pixels_out = np.transpose(pixels_out, (0, 3, 1, 2))

        # add the image to the array
        pix_in = np.append(pix_in, pixels_in, axis=0)
        pix_out = np.append(pix_out, pixels_out, axis=0)

        if cnt % batch == 0:
            # convert the input array into a tensor
            pix_in = torch.tensor(pix_in).float()
            pix_out = torch.tensor(pix_out).float()

            # run the lpips evaluation algorithm
            res = lpips.forward(pix_in, pix_out)
            sum_lpip = sum_lpip + torch.mean(res).item()

            #empty the image array to prepare to fill it with the next batch
            pix_in = np.empty([0, 3, 256, 256])
            pix_out = np.empty([0, 3, 256, 256])

            # increment the number of lpips evals that have been gathered
            comps += 1
        cnt += 1

    if comps == 0:
        return 0
    return sum_lpip/comps

