import os

import torch
import torch.nn as nn 
from torch.optim import Adam

import numpy as np 

import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import seaborn as sns 
from IPython.display import clear_output

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

#---------------------------------
#      Trajectory simulator
#---------------------------------

class NaviTraj:

    def __init__( self, seed=42):
        '''2D straight-line navigaqtion trajectory generator

        Reference: 
        https://www.pnas.org/content/suppl/2021/12/16/2018422118.DCSupplemental

       
        Sensory input x:

        x = H( W_in @ x.T)

        W_in: This l is projected to a random projector W_in 300 x 6 matrix
            Winp[ i, j] ~ N( 0, 1). The
        H: a step-wise function return a binary input 

        '''
        self.rng = np.random.RandomState( seed)
        self.W_in = self.rng.randn( 300, 6)
        self.H    = lambda x: 1. * ( x > 0)
        self.reset()

    def reset( self):
        '''Init the wall location 
        '''
        self.phi   = self.rng.uniform( 0, 2*np.pi)
        self.theta = 0 
        self.d     = 0 

    def towards( self):
        '''Heading from the wall
        '''
        self.theta = self.rng.uniform( -np.pi/2, np.pi)

    def step( self, t):
        '''step fowards
        
        Any point s in the 2D space can be decided by 3-tuple
        phi: the starting position. Defined 
            as the angle between a fixed direction(East)
        theta: the moving direction. Defined as 
            the angle with respect to the starting wall
        d: the distance travelled since the last wall contact.

        s = [ cos(2πd/sqrt(8)), sin(2πd/sqrt(8)),
              cos(theta), sin(theta),
              cos(phi),   sin(phi) ]
        '''
        d = .1 * t
        return np.array( [ np.cos( 2*np.pi*d / np.sqrt(8)), np.sin( 2*np.pi*d / np.sqrt(8)),
                           np.cos( self.theta), np.sin( self.theta),
                           np.cos( self.phi), np.sin( self.phi)])

    def rollout( self, N=500, Verbose=False):
        '''Generate the true trajectory

        Input:
            N: number of trajectory
            Verbose: visualize the environment of not 

        Each session contains 500 trajectories.
        For each trajectory:
            An initial position is sampled.
                phi ~ Uniform( 0, 2*pi)
            A heading function is sampled
                theta ~ [ -pi/2, pi/2]
            While not reach wall:
                Increase the distance
                    d += 1 
        '''
        # Storages
        traj = [] 

        # repeat 500 trajectories 
        for _ in range(N):
            # sample the heading direction
            done, t = False, 0
            self.towards()
            # move ahead util reach the wall
            while not done: 
                t += 1
                traj.append( self.step( t))
                # visualize the env
                if Verbose:
                    clear_output(True)
                    room = Rectangle( (-1, -1), 
                                        2,  2,
                                        linewidth=1,
                                        edgecolor='r',
                                        facecolor='none')
                # check if reach the wall 
                if self.hitwall():
                    done = True 
        
        return np.vstack( traj)

    def state2obs( self, state):
        '''State to observation
            Shape: [?. 6] x [ 6, 300] = [?, 300] 
        '''
        return self.H( state @ self.W_in.T)
        
#--------------------------------
#        Hyperparameters
#--------------------------------

DefaultParams = { 
                'MaxEpochs': 15,
                'L2Reg': 1e-5,
                'SparsityReg': 5,
                'SparsityPro': .1,
                'Verbose': True,
                'BatchSize': 32,
                'If_gpu': True, 
                } 
eps_ = 1e-8

#---------------------------------
#    AutoEncoder architecture
#---------------------------------

class AE( nn.Module):

    def __init__(self, dims, gpu=False):
        super().__init__()
        # choose device 
        if gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        # construct encoder 
        encoder = [] 
        for i in range(len(dims)-1):
            encoder.append( nn.Linear( dims[i], dims[i+1]))
            encoder.append( nn.Sigmoid())
        self.encoder = nn.Sequential( *encoder).to(self.device)
        # construct decoder, reverse the encoder operation
        re_dims = list(reversed(dims))
        decoder = [] 
        for i in range(len(dims)-1):
            decoder.append( nn.Linear( re_dims[i], re_dims[i+1]))
            decoder.append( nn.Sigmoid())
        self.decoder = nn.Sequential( *decoder).to(self.device)
        
    def forward( self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def encode( self, x):
        '''Encode for test only
        '''
        with torch.no_grad():
            return self.encoder(x)

    def decode( self, z):
        '''Decode for test only
        '''
        with torch.no_grad():
            return self.decoder(z)

    def MSEsparse( self, x, x_hat, z, spa_reg=5, p=.1):
        '''Mean sqaure error sparse

        Error = 1/N ∑_i∑_j (x_ij - x_hat_ij)^2  
                    + β1 l2norm(W)
                    + β2 Sparity
        
        Note that l2 regularize can be implemented
        in the optimizer using weight decay:
        https://pytorch.org/docs/stable/optim.html

        '''
        mse = (x - x_hat).pow(2).sum(1).mean()
        sparsity = self.SparsityLoss( z, p)
        return mse + spa_reg*sparsity

    def SparsityLoss( self, z, p_tar):
        '''
        The supplementary material of the Benna and Fusi 2021 says:
            "The level of activate is esimated by binarized the 
                latent representation by the threshold = 0.5"

        We use the binary value to estimate the activate 
        proportion: p_hat  = 1/N ∑_i Z_i(x)

        Reference
        https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
        '''
        # get the target tesnor 
        rho_hat = z.mean(dim=1)
        rho     = (torch.ones_like( rho_hat) * p_tar).to(self.device)

        # calculate kld 
        kld1 = rho * ((rho).log() - (rho_hat).log())
        kld2 = (1-rho) * ((1-rho).log() - (1-rho_hat).log())
        return (kld1 + kld2).sum()

#---------------------------
#    Train Auto encoder 
#---------------------------

def trainAE( train_data, dims, **kwargs):
    '''Train a sparse autoencoder

    Input:
        data: the data for training
        dims: the dimension for your network architecture
        L2Reg: the weight for l2 norm
        SparsityReg: the weight for sparsity
        SparsityPro': the target level sparsity 
        MaxEpochs: maximum training epochs 
        BatchSize: the number of sample in a batch for the SGD
        Versbose: tracking the loss or ont 
    '''
    ## Prepare for the training 
    # set the hyper-parameter 
    HyperParams = DefaultParams
    for key in kwargs.keys():
        HyperParams[key] = kwargs[key]
    # preprocess the data
    x, y = train_data
    n_batch = int( len(x))
    x_tensor = x.type( torch.FloatTensor)
    y_tensor = y.type( torch.FloatTensor)
    _dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    _dataloader = torch.utils.data.DataLoader( _dataset, 
                batch_size=HyperParams['BatchSize'], drop_last=True)
    # init model
    model = AE( dims, gpu=HyperParams['If_gpu'])
    # decide optimizer 
    optimizer = Adam( model.parameters(), lr=1e-4, 
                        weight_decay=HyperParams['L2Reg'])       
    ## get batch_size
    losses = []
    
    # start training
    model.train()
    for epoch in range( HyperParams['MaxEpochs']):

        ## train each batch 
        loss_ = 0        
        for i, (x_batch, _) in enumerate(_dataloader):

            # reshape the image
            x = torch.FloatTensor( x_batch).view( 
                x_batch.shape[0], -1).to( model.device)
            # reconstruct x
            z, x_hat =  model.forward( x)
            # calculate the loss 
            loss = model.MSEsparse( x, x_hat, z, 
                spa_reg=HyperParams['SparsityReg'],
                p=HyperParams['SparsityPro'])
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # store the te losses for plotting 
            loss_ += loss.detach().cpu().numpy() / n_batch

        # track training
        losses.append(loss_)
        if (epoch%1 ==0) and HyperParams['Verbose']:
            print( f'Epoch:{epoch}, Loss:{loss_}')

    return model, losses 

if __name__ == '__main__':

    # Simulate trajectory
    rat = NaviTraj(seed=2022)
    rat.rollout()