import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


tensorboard_writer = SummaryWriter()
print(tensorboard_writer.get_logdir())


# function to convert matplotlib fig to numpy
def fig2data(fig):

    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


# generate random points
size = 500
x = np.random.uniform(0, 2., size=500)
y = np.random.uniform(0, 2., size=500)
trajectory_len = len(x)
trajectory_indices = np.arange(trajectory_len)

# figure dimensions
width, height = 3, 2

# tensorboard takes video of shape (B,C,T,H,W)
video_array = np.zeros(
    shape=(1, 3, trajectory_len, height*100, width*100),
    dtype=np.uint8)

# plot each point, one at a time
for trajectory_idx in trajectory_indices:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width, height),
        gridspec_kw={'width_ratios': [1, 0.05]})
    fig.suptitle('Example Trajectory')
    # plot the first trajectory
    sc = axes[0].scatter(
        x=[x[trajectory_idx]],
        y=[y[trajectory_idx]],
        c=[trajectory_indices[trajectory_idx]],
        s=4,
        vmin=0,
        vmax=trajectory_len,
        cmap=plt.cm.jet)

    axes[0].set_xlim(-0.25, 2.25)
    axes[0].set_ylim(-0.25, 2.25)

    colorbar = fig.colorbar(sc, cax=axes[1])
    colorbar.set_label('Trajectory Index Number')

    # extract numpy array of figure
    data = fig2data(fig)

    # UNCOMMENT IF YOU WANT TO VERIFY THAT THE NUMPY ARRAY WAS CORRECTLY EXTRACTED
    # plt.show()
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, frameon=False)
    # ax2.imshow(data)
    # plt.show()

    # close figure to save memory
    plt.close(fig=fig)

    video_array[0, :, trajectory_idx, :, :] = np.transpose(data, (2, 0, 1))

# tensorboard takes video_array of shape (B,C,T,H,W)
tensorboard_writer.add_video(
    tag='sampled_trajectory',
    vid_tensor=torch.from_numpy(video_array),
    global_step=0,
    fps=4)

print('Added video')

tensorboard_writer.close()

video_array = np.zeros(
    shape=(1, trajectory_len, 3, height*100, width*100),
    dtype=np.uint8)
video_array[0, trajectory_idx, :, :, :] = np.transpose(data, (2, 0, 1))