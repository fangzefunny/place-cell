import os
from sre_parse import Verbose
import torch
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from utils.autoencoder import trainAE

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')

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

#----------------------------
#     Correlation metric  
#---------------------------

'''Unknown metric to us

Need more information from the paper
'''
corr = lambda x: x

#----------------------------
#     Ultrametric trees 
#---------------------------

def UltrametricTree_data( M, N, k, gamma=.2, rng=None, anc=None):
    '''Create Ultrametric tree data
    Input: 
        M: the number of data (rows)
        N: the size of the data (columns)
        k: the branch size of the tree 
        gamma: the rate of a flipping branch
        rng: random state generator
    '''
    # get random generate 
    rng = rng if rng else np.random.RandomState(42)
    # init storage
    sim_data = np.zeros( [ M, N]) + np.nan
    # decide number of ancestor 
    p = int(N / k)
    # get descent: data 
    anc = (rng.rand(p) < .5 ) if anc is None else anc 
    for m in range(M):   
        sim = []
        for i in anc:
            for _ in range(k):
                d = i if rng.rand() < gamma else 1-i
                sim.append( d)
        sim_data[ m, :] = sim
    return sim_data

def fig_2b( seed=2020):
    '''Intra class corr
        Ignore the random encoder 
    '''
    ## fix random seed 
    rng = np.random.RandomState( seed)
    ## k list  
    ks = [ 10, 20, 30, 60, 75, 100, 150, 300]
    dims = [ 300, 600]
    IN_corrs = []
    AE_corrs = []
    M, N = 2000, 300
    gamma = .6
    for k in ks: 
        # decide the ancestor
        # decide number of ancestor 
        p = int(N / k)
        # get descent: data 
        anc = (rng.rand(p) < .5 ) * 1.
        # generate input data 
        sim_data = UltrametricTree_data( M, N, k, gamma, rng=rng, anc=anc)
        train_data = torch.FloatTensor( sim_data)
        train_label = train_data # not used, just a placeholder to use dataloader
        # train an AE
        model = trainAE( (train_data, train_label), dims, 
                            SparsityReg=3, SparistyPro=.1,
                            If_gpu=True, MaxEpochs=600)
        # calculate the correlation
        AE_corr = []
        for _ in  range(10*M):
            test_data = UltrametricTree_data( 2, N, k, gamma, rng=rng, anc=anc)
            model.eval()
            z = model.encoder( torch.FloatTensor(test_data)
                    ).detach().cpu().numpy()
            AE_corr.append( corr( z[0,:], z[1,:])
        #AE_corrs.append( np.mean(AE_corr))
    print( IN_corrs)
    print( AE_corrs)
    plt.figure( figsize=( 4, 4))
    plt.plot( ks, IN_corrs, 'o-', color=Red)
    plt.plot( ks, AE_corrs, 'o-', color=Blue)
    plt.legend( [ 'input', 'autoencoder'], fontsize=fontsize)
    plt.xlabel( 'Braching ratio', fontsize=fontsize)
    plt.ylabel( 'Intra class corr.', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig( f'{path}/figures/fig_2b', dpi=dpi)

if __name__ == '__main__':

    ## Get Mnist data 
    

    ## Compress 
    dims = [ 784, 196]
    for f in [ 0, 3]:

        ## Load a model. If no model, train one 
        try:
            model = AE( dims, gpu=False)
            model.load_state_dict(torch.load(f'{path}/checkpts/mnist_model-f={f}.pkl'))
        except:
            print( f'Train AE with sparisty={f}')
            model, losses = trainAE( (data, label), dims, SparsityReg=f)
            torch.save( model.state_dict(), f'{path}/checkpts/mnist_model-f={f}.pkl')

        ## Visualize
        model.to('cpu') 
        model.eval()
        rng = np.random.RandomState( 2022)
        ind = rng.choice( data.shape[0], size=10)
        reconstruct( data, model, ind=ind, f=f)
        ind = rng.choice( 196, size=25)
        decode_Z( model, 196, ind=ind, f=f)

