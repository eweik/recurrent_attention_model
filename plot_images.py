'''
plot_images.py

Recurrent Attention Model (RAM) Project
PyTorch
'''
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import torch
from torchvision import datasets
from torchvision import transforms

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str, required=True,
                     help="path to directory containing pickle dump")
    arg.add_argument("--epoch", type=int, required=True,
                     help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch']
    

def bounding_box(x, y, size, color='w'):
    x = int(x - (size/2))
    y = int(y - (size/2))
    rect = patches.Rectangle((x,y), size, size, 
                             linewidth=1,
                             edgecolor=color,
                             fill=False)
    return rect

def denormalize(L, coords):
    return ((coords + 1.0) * L) / 2.
    

def plot_base_image(axs, glimpses):
    for i, ax in enumerate(axs.flat):
        ax.imshow(glimpses[i], cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    return axs
    

def plot_glimpses(plot_dir, epoch):
    glimpses = pickle.load( open(plot_dir+"g_{}.p".format(epoch), "rb") )
    locations = pickle.load( open(plot_dir+"l_{}.p".format(epoch), "rb") )
    glimpse = np.concatenate(glimpses)
    
    # gramp useful parameters
    size = int( plot_dir.split('_')[2][0] )
    num_anims = len(locations)
    num_cols = glimpses.shape[0]
    img_shape = glimpses.shape[1]
    
    # denormalize coordinates
    coords = [denormalize(img_shape, c) for c in locations]
    
    # plot base image
    fig, axs = plt.subplots(nrows=1, ncols=num_cols)
    axs = plot_base_image(axs, glimpses)
        
    def updateData( img ):
        color = 'r'
        co = coords[i]
        for i, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[i]
            rect = bounding_box(c[0], c[1], size, color)
            ax.add_patch(rect)
    
    # animate
    anim = animation.FuncAnimation(fig, 
                                   updateData, 
                                   frames=num_anims,
                                   interval=500,
                                   repeat=True)
    
    # save as mp4
    name = plot_dir + "epoch_{}.mp4".format(epoch)
    anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
    return
    

def plot_images(images, gd_truth):

    images = images.squeeze()
    assert len(images) == len(gd_truth) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = "{}".format(gd_truth[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    return


def get_dataloader(data_dir):
    dataset = datasets.MNIST(data_dir, 
                             train=True, 
                             download=True, 
                             transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=9, 
                                             shuffle=True)
    return dataloader                        


def show_sample(data_dir):
    dataloader = get_dataloader(data_dir)
    data_iter = iter(dataloader)
    images, labels = data_iter.next()
    X = images.numpy()
    X = np.transpose(X, [0, 2, 3, 1])
    plot_images(X, labels)
    return







