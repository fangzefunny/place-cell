import os
import sys 
import torch
from torchvision import datasets, transforms
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from utils.autoencoder import AE, trainAE

#--------------------------------
#        System variables
#--------------------------------

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')
if not os.path.exists(f'{path}/checkpts'):
    os.mkdir(f'{path}/checkpts')

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors    = [ Blue, Red, Green, Yellow, Purple]
sns.set_style("whitegrid", {'axes.grid' : False})

# image dpi
dpi = 250
fontsize = 16

# get scale 
img_scale  = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

#--------------------------------
#        Visualization 
#--------------------------------

def reconstruct( data, model, ind=range(10), f=0):
    
    nr = 2
    nc = len( ind) 
    fig, axs = plt.subplots( nr, nc, figsize=( nc*2.5, nr*2.5))
    for i in range(nc):
        img = data[ ind[i], :, :]
        _, img_hat =  model( torch.FloatTensor(img.reshape([1,-1])))
        # draw the original image on the left
        ax = axs[ 0, i]
        ax.imshow( img, cmap='gray', vmin=0, vmax=1)
        ax.set_title( f'Raw Image: {ind[i]}')
        ax.set_axis_off()
        # draw the reconstructed image in the middel 
        ax = axs[ 1, i]
        ax.imshow( img_hat.detach().cpu().numpy().reshape( [ 28, 28]), cmap='gray', vmin=.2, vmax=.8)
        ax.set_title( f'Reconstructed Image: {ind[i]}')
        ax.set_axis_off()
        
    fig.tight_layout()
    plt.savefig( f'{path}/figures/Mnist_recon-f={f}.png')

def decode_Z( model, z_dim, ind=range(25), f=0):

    nr = 5
    nc = 5 
    z = np.zeros( [ len(ind), z_dim])
    for i, idx in enumerate(ind):
        z[ i, idx] = 1.
    img_hat =  model.decode( torch.FloatTensor(z)
            ).detach().cpu().numpy().reshape( [ len(ind), 28, 28])
    fig, axs = plt.subplots( nr, nc, figsize=( nc*2.5, nr*2.5))
    for i, idx in enumerate(ind):
        ax = axs[ i//nr, i%nr]
        ax.imshow(img_hat[ i, :, :], cmap='gray', vmin=.2, vmax=.8)
        ax.set_title( f'{idx}th Latent layers')
        ax.set_axis_off()
     
    fig.tight_layout()
    plt.savefig( f'{path}/figures/Mnist_decode-f={f}.png')

if __name__ == '__main__':

    ## Get Mnist data 
    mnist_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transforms.Compose(
                                    [ transforms.ToTensor(), transforms.Normalize( (.1307,), (.3081,))]
                                ))
    data = (mnist_data.data.type( torch.FloatTensor) / 255).bernoulli()
    label = (mnist_data.targets.type( torch.FloatTensor) / 255).bernoulli()

    ## Compress 
    dims = [ 784, 196]
    for f in [ 0, 3]:

        ## Load a model. If no model, train one 
        try:
            model = AE( dims, gpu=False)
            model.load_state_dict(torch.load(f'{path}/checkpts/mnist_model-f={f}.pkl'))
        except:
            print( f'Train AE with sparisty={f}')
            model, losses = trainAE( (data, label), dims, SparsityReg=f, if_gpu=True)
            torch.save( model.state_dict(), f'{path}/checkpts/mnist_model-f={f}.pkl')

        ## Visualize
        model.to('cpu') 
        model.eval()
        rng = np.random.RandomState( 2022)
        ind = rng.choice( data.shape[0], size=10)
        reconstruct( data, model, ind=ind, f=f)
        ind = rng.choice( 196, size=25)
        decode_Z( model, 196, ind=ind, f=f)


