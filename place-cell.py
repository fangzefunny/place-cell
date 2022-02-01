import os
from tabnanny import verbose
import scipy as sp
import torch
import numpy as np 
from scipy.signal import convolve2d

import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import seaborn as sns

from utils.autoencoder import AE, trainAE

'''
Need to do:

    Decode the latent layer.
'''

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

#-------------------------------
#        Axullary functions 
#-------------------------------

# get scale 
img_scale  = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

# max N 
argmaxN = lambda arr, n: (-arr).argsort()[:n]

def loc2phi( loc):
    '''Angle between (1, 0)
    phi = arccos( <u, v>/||u||||v||)
    '''
    unit_loc = loc / np.linalg.norm(loc)
    eign_vec = np.array( [ 1, 0])
    dot_product = np.dot( unit_loc, eign_vec)
    return np.arccos(dot_product)

def to_grid( loc):
    loc_abs = loc - np.array( [ -1, -1])
    return ( loc_abs * 10).astype(int)

def gaussFilter(size=3, sigma=1):
    '''2D Gaussian filter

    The paper says one bin gauss filter
    should be with size = 3x3, 
    but the sigma is not told, let's say 1 
    '''
    n = (size-1)//2
    x, y = np.mgrid[-n:n+1, -n:n+1]
    g = np.exp(-((x**2 + y**2)/(2.*sigma**2)))
    return g/g.sum()

def smooth( tuning):
    '''Convolve with a gaussian filter
    '''
    filter = gaussFilter()
    return convolve2d( tuning, filter, mode='valid')

    
#---------------------------------
#      Trajectory simulator
#---------------------------------

class Sensory:

    def __init__( self, seed=42):
        '''Get sensory input x from state

        x = H( W_in @ x.T)

        W_in: This l is projected to a random projector W_in 300 x 6 matrix
            Winp[ i, j] ~ N( 0, 1). The
        H: a step-wise function return a binary input 
        '''
        self.rng = np.random.RandomState( seed)
        self.W_in = self.rng.randn( 300, 6)
        self.H    = lambda x: 1. * ( x > 0)
    
    def state2obs( self, state):
        '''State to observation
            Shape: [?. 6] x [ 6, 300] = [?, 300] 
        '''
        return self.H( state @ self.W_in.T)

class NaviTraj:

    def __init__( self, seed=42):
        '''2D straight-line navigaqtion trajectory generator

        Reference: 
        https://www.pnas.org/content/suppl/2021/12/16/2018422118.DCSupplemental


        Note that due to the lack of detailed information, especially
        how to decide whether it hits the wall, I use an approxiamte 
        method to construct the navigation trajectory.

        '''
        self.rng = np.random.RandomState( seed)

    def reset( self):
        '''Init the wall location 
        '''
        # this method if quite different from what
        # the paper says φ ~ Uni(0, 2pi)
        pos = self.rng.uniform( 0, 8)
        if pos < 2:
            loc = [ -1+pos, -1] # South ( x, -1)
            self.wallortho = np.pi/2
        elif pos < 4:
            loc = [  1, -3+pos] # East  (  1, x)
            self.wallortho = np.pi
        elif pos < 6:
            loc = [ -5+pos,  1] # North ( x,  1)
            self.wallortho = 3 * np.pi/2
        else:
            loc = [ -1, -7+pos] # West  ( -1, x)
            self.wallortho = 0
        self.loc   = np.array( loc)
        self.phi   = loc2phi( self.loc)
        self.theta = self.rng.uniform( -np.pi/2, np.pi/2)
        self.d     = 0 

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
        ang = self.wallortho + self.theta
        if t > 0:
            self.loc += np.array( [ .1*np.cos( ang), .1*np.sin(ang)])
        return self.loc, np.array( [ np.cos( 2*np.pi*d / np.sqrt(8)), np.sin( 2*np.pi*d / np.sqrt(8)),
                           np.cos( self.theta), np.sin( self.theta),
                           np.cos( self.phi), np.sin( self.phi)])

    def rollout( self, sess, N=500, Verbose=False):
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
        X, Y = [], [] 
        meshgrid = { i: { j: [] for j in range(21)} for i in range(21)}
        m2 = { i: { j: [] for j in range(21)} for i in range(21)}

        # repeat 500 trajectories
        if verbose: 
            fig, ax = plt.subplots( 1, 1, figsize=(5,5))
        for _ in range(N):
            # sample the heading direction
            t = 0
            self.reset()
            while True: 
                loc, state = self.step(t)
                if (t>0) and (self.hit_wall()):
                    X.append( np.nan)
                    Y.append( np.nan)
                    break 
                cat = to_grid( loc)
                # note that put the col idx 
                # prior to the row idx because
                # a two dim matrix is a transpose
                meshgrid[cat[1]][cat[0]].append( state.copy())
                m2[cat[1]][cat[0]].append( loc.copy())
                if any(abs(loc) > 1):
                    print(1)
                X.append( loc[0])
                Y.append( loc[1])
                traj.append( state)
                if Verbose:
                    self.render( ax, X, Y)
                t += 1

        # save trajectories
        # plt.savefig( f'figures/traj-{sess}.png') 

        return np.vstack(traj), meshgrid

    
    def hit_wall( self):
        return np.max(abs(self.loc))>=1

    def render(self, ax, X, Y):
        ax.plot( X, Y, color=Red)
        room = Rectangle( (-1.01, -1.01), 
                            2.02,  2.02,
                            linewidth=2,
                            edgecolor=Blue,
                            facecolor='none')
        ax.add_patch( room)
        ax.set_axis_off()

#--------------------------------
#         Spatial tuning 
#--------------------------------

def spatial_field( test_data, model, sensor, z_dim):
    '''Spatial tuning analysis

    The supplementary material of the Benna and Fusi 2021 says:
        - "we construct maps of the cumulative activity of 
        individual hidden-layer units"
        - "normalize them by the total occupancy of each 
        21x 21 spatial bins." 
        - "smooth them by convoling a Gaussian filter of width one bin"
        - 
    '''
    spa_tuning = np.zeros( [ 21, 21, z_dim])
    for i in test_data.keys():
        for j in test_data[i].keys():
            if len(test_data[i][j]):
                in_data = np.vstack( test_data[i][j])
                x = sensor.state2obs( in_data)
                z = model.encode( torch.FloatTensor(x)
                    ).detach().cpu().numpy().sum(0)
                spa_tuning[ i, j, :] = z 
    return spa_tuning / spa_tuning.sum(2, keepdims=True)

def spatial_tuning( field_all, ind):
    '''Spatial tuning analysis
    '''
    ## Setup some hyper values
    nr = nc = 6 
    fig, axs = plt.subplots( nr, nc, figsize=( nc*2.5, nr*2.5))
    ind = argmaxN( field_all.sum(1).sum(0), 36)
    for i, idx in enumerate(ind):
        # get cumulative activity of each units
        tuning = field_all[ :, :, idx]
        # convolve with a one bin gaussian filter
        tuning_smooth = smooth( tuning)
        ax = axs[ i//nr, i%nr]
        ax.imshow( tuning_smooth, cmap='viridis')
        ax.set_title( f'{idx}th Latent layers')
        ax.set_axis_off()
    fig.tight_layout()
    plt.savefig( f'{path}/figures/Spa_tuning.png')

if __name__ == '__main__':

    # Simulate trajectory
    seed = 2022
    
    ## Compress 
    dims = [ 300, 600]
    train = False
    tot_Sess = 60
    sensor = Sensory( seed)
    ind = np.random.RandomState(seed).choice( dims[-1], size=36)
    model = AE( dims, gpu=True)

    #Load a model. If no model, train one 
    if train:
        for s in range(tot_Sess):
            seed += 1 
            print( f'Train AE @ Session: {s}; Seed:{seed}')
            traj, _  = NaviTraj( seed).rollout( s, N=500, Verbose=False)
            data  = torch.FloatTensor( sensor.state2obs( traj))
            label = torch.FloatTensor( traj) #just a placeholder to use the dataloader
            model, losses = trainAE( (data, label), model, LR=1e-3,
                                        SparsityReg=1, SparsityPro=.03,
                                        L2Reg=0, if_gpu=True, BatchSize=32,
                                        MaxEpochs=15)
            if s >=19:
                # "excluding the transient learning period that occurs during the first 20 sessions"
                torch.save( model.state_dict(), f'{path}/checkpts/traj_model-S{s}.pkl')
                
    ## Visualize
    field_all = np.zeros( [ 21, 21, dims[-1]])
    seed = 2022 + 19
    for s in range( 19, tot_Sess):  
        seed += 1
        print( f'Test AE @ Session: {s}; Seed:{seed}')  
        model = AE( dims, gpu=False)
        model.load_state_dict(torch.load(f'{path}/checkpts/traj_model-S{s}.pkl'))
        model.to('cpu') 
        model.eval()
        # spatial tunning property
        _, spa  = NaviTraj( seed).rollout( s, N=500, Verbose=False)
        field_all += spatial_field( spa, model, sensor, dims[-1]) / 40

    # average over trial
    spatial_tuning( field_all, ind)
            
            
