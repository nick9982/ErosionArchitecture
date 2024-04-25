from matplotlib import pyplot
from data import read_file_into_list
import numpy as np
from lpips.lpips import LPIPS

def graphGANLoss():
    dL1_ls = read_file_into_list('dL1_ls.txt')
    dL2_ls = read_file_into_list('dL2_ls.txt')
    iterations_loss = read_file_into_list('loss_iterations.txt')

    pyplot.cla() 
    pyplot.axis("on")

    pyplot.figure()
    pyplot.plot(iterations_loss, dL1_ls, label='DL1 Loss', color='r')
    pyplot.plot(iterations_loss, dL2_ls, label='DL2 Loss', color='b')

    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss')
    pyplot.title('Loss')

    pyplot.legend(loc='upper right')
    pyplot.savefig('loss.png')
    pyplot.close()

def saveLPIPScores():
    lpips_val_scores = read_file_into_list('lpips_val.txt')
    lpips_training_scores = read_file_into_list('lpips_training.txt')
    lpips_iterations = read_file_into_list('lpips_iterations.txt')
    pyplot.cla()
    pyplot.axis("on")

    pyplot.figure()
    pyplot.plot(lpips_iterations, lpips_val_scores, label='val', color='b')
    pyplot.plot(lpips_iterations, lpips_training_scores, label='training', color='r')

    pyplot.xlabel('Iteration')
    pyplot.ylabel('lpips')
    pyplot.title('lpips')

    pyplot.legend(loc='upper right')

    pyplot.savefig('lpips.png')

def plotMAE():
    iterations = read_file_into_list('lpips_iterations.txt')
    mae = read_file_into_list('mae.txt')
    mae_val = read_file_into_list('mae_val.txt')
    pyplot.cla()
    pyplot.figure()
    pyplot.plot(iterations, mae_val, label='val', color='b')
    pyplot.plot(iterations, mae, label='training', color='r')
    pyplot.legend(loc='upper right')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('mae')
    pyplot.title('MAE Scores')

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

    pyplot.close()
    # save the generator model
    filename2 = "model_%06d.h5" % (step + 1)
    g_model.save(filename2)

def mae(real_dir, fake_dir):
    files = listdir(real_dir)
    sm = 0
    cnt = 0 
    for real in files:
        fake = fake_dir + real.replace('real', 'fake')
        pixels_in = imageio.imread(real_dir + real)
        pixels_out = imageio.imread(fake)

        pixels_in = image.img_to_array(pixels_in)
        pixels_out = image.img_to_array(pixels_out)

        pixels_in = image.img_to_array(pixels_in)
        pixels_out = image.img_to_array(pixels_out)

        mae_loss = float(tf.reduce_mean(tf.abs(pixels_in - pixels_out)).numpy())
        sm += mae_loss
        cnt += 1

    return mae_loss/cnt

def lpips_eval(real_dir, fake_dir):
    files = listdir(real_dir)
    sum_lpip = 0
    pix_in = np.empty([0, 3, 256, 256])
    pix_out = np.empty([0, 3, 256, 256])
    batch = 64
    cnt = 1
    comps = 0
    for real in files:
        fake = fake_dir + real.replace('real', 'fake')
        pixels_in = imageio.imread(real_dir + real)
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

