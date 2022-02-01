import os
import torch
import numpy as np 

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

def flat_meshgrid( mesh):
    traj = [] 
    for i in mesh.keys():
        for j in mesh[i].keys():
            traj.append( np.array(mesh[i][j]).reshape([-1,6]))
    return np.concatenate( traj, axis=0)

def norm_discrete( x, bins=21*21):
    '''
    '''
    x_dn = np.histogram( x, bins=bins)[0] / bins 
    n = int(np.sqrt(bins))
    return x_dn.reshape( [n, n])
    
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

        Note that due to the lack of detailed information, especially
        how to decide whether it hits the wall, I use an approxiamte 
        method to construct the navigation trajectory.

        '''
        self.rng = np.random.RandomState( seed)
        self.W_in = self.rng.randn( 300, 6)
        self.H    = lambda x: 1. * ( x > 0)

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
        self.loc   = loc
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

    def rollout( self, N=500, Verbose=False, mode='train'):
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

        # repeat 500 trajectories 
        fig, ax = plt.subplots( 1, 1, figsize=(5,5))
        for _ in range(N):
            # sample the heading direction
            done, t = False, 0
            self.reset()
            while not done: 
                loc, state = self.step(t)
                if (t>0) and (self.hit_wall()):
                    X.append( np.nan)
                    Y.append( np.nan)
                    break 
                cat = to_grid( loc)
                # note that put the col idx 
                # prior to the row idx because
                # of python's reshape direction is row 
                meshgrid[cat[1]][cat[0]].append( state)
                X.append( loc[0])
                Y.append( loc[1])
                traj.append( state)
                if Verbose:
                    self.render( ax, X, Y)
                t += 1

        # save trajectories
        self.render( ax, X, Y)
        plt.savefig( f'figures/trajectories.png')
        if mode == 'train':
            return np.vstack(traj)
        elif mode == 'test':
            return meshgrid
    
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

    def state2obs( self, state):
        '''State to observation
            Shape: [?. 6] x [ 6, 300] = [?, 300] 
        '''
        return self.H( state @ self.W_in.T)

#--------------------------------
#        Visualization 
#--------------------------------

def spatial_tuning( data, model, z_dim, seed=42):
    '''Spatial tuning analysis

    The supplementary material of the Benna and Fusi 2021 says:
        - "we construct maps of the cumulative activity of 
          individual hidden-layer units"
        - "normalize them by the total occupancy of each 
          21x 21 spatial bins." 
        - "smooth them by convoling a Gaussian filter of width one bin"
        - "excluding the transient learning period that occurs during
           the first 20 sessions"
    '''
    ## Setup some hyper values
    nr = nc = 6 
    rng = np.random.RandomState(seed)
    ind = rng.choice( z_dim, size=nr*nc)
    
    fig, axs = plt.subplots( nr, nc, figsize=( nc*2.5, nr*2.5))
    for i, idx in enumerate(ind):
        ## Get cumulative activity of each units 
        z, _ = model( data)
        latent = z.detach().cpu().numpy()
        # get cumulative data 
        spatial_z = norm_discrete( latent[ :, idx])
        # 
        ax = axs[ i//nr, i%nr]
        ax.imshow(spatial_z)
        ax.set_title( f'{idx}th Latent layers')
        ax.set_axis_off()

    fig.tight_layout()
    plt.savefig( f'{path}/figures/Spa_tuning.png')

if __name__ == '__main__':

    # Simulate trajectory
    navi  = NaviTraj( seed=2022)
    traj  = navi.rollout( N=500, Verbose=False)
    data  = torch.FloatTensor( navi.state2obs( traj))
    label = torch.FloatTensor( traj) #just a placeholder to use the dataloader
    
    ## Compress 
    dims = [ 300, 600]
    #Load a model. If no model, train one 
    try:
        model = AE( dims, gpu=False)
        model.load_state_dict(torch.load(f'{path}/checkpts/traj_model.pkl'))
    except:
        print( f'Train AE....ing')
        model, losses = trainAE( (data, label), dims, 
                                    SparsityReg=1, SparsityPro=.03,
                                    L2Reg=0, if_gpu=True)
        torch.save( model.state_dict(), f'{path}/checkpts/traj_model.pkl')

    ## Visualize
    # speed up by turning on the test mode
    model.to('cpu') 
    model.eval()
    # spatial tunning property
    navi  = NaviTraj( seed=2022)
    traj  = navi.rollout( N=500, Verbose=False, mode='train')
    test_data  = torch.FloatTensor( navi.state2obs( traj))
    spatial_tuning( test_data, model, navi, 600, seed=2020)
