import torch
import torch.nn as nn 
from torch.optim import Adam
from tqdm import tqdm 
import numpy as np 

DefaultParams = { 
                'MaxEpochs': 600,
                'L2Reg': 1e-5,
                'SparsityReg': 5,
                'SparsityPro': .1,
                'Verbose': False,
                } 
eps_ = 1e-8
#----------------------------
#   Loss functions 
#----------------------------

def MSEsparse( x, x_hat, z, spa_reg=5, p=.1):
    '''Mean sqaure error sparse

    Error = 1/N ∑_i∑_j (x_ij - x_hat_ij)^2  
                + β1 l2norm(W)
                + β2 Sparity
    
    Note that l2 regularize can be implemented
    in the optimizer using weight decay:
    https://pytorch.org/docs/stable/optim.html

    '''
    mse = (x - x_hat).pow(2).mean()
    sparsity = SparsityLoss( z, p)
    return mse + spa_reg*sparsity

def SparsityLoss( z, p_tar, thershold=.5):
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
    rho_hat = torch.sigmoid(z).mean(dim=1)
    rho     = torch.ones_like( rho_hat) * p_tar

    # calculate kld 
    kld1 = rho * ((rho+eps_).log() - (rho_hat+eps_).log())
    kld2 = (1-rho) * ((1-rho+eps_).log() - (1-rho_hat+eps_).log())
    return (kld1 + kld2).mean()

#---------------------------------
#    AutoEncoder architecture
#---------------------------------

class AE( nn.Module):

    def __init__(self, dims):
        super().__init__()
        # construct encoder 
        encoder = [] 
        for i in range(len(dims)-1):
            encoder.append( nn.Linear( dims[i], dims[i+1]))
            encoder.append( nn.Sigmoid())
        self.encoder = nn.Sequential( *encoder)
        # construct decoder, reverse the encoder operation
        re_dims = list(reversed(dims))
        decoder = [] 
        for i in range(len(dims)-1):
            decoder.append( nn.Linear( re_dims[i], re_dims[i+1]))
            decoder.append( nn.Sigmoid())
        self.decoder = nn.Sequential( *decoder)

    def forward( self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def get_latent( self, x):
        x = torch.FloatTensor( x).view( x.shape[0], -1)
        return self.encoder(x).detach().numpy()


#---------------------------
#    Train Auto encoder 
#---------------------------

def trainAE( data, dims, **kwargs):
    '''Train a sparse autoencoder

    Input:
        data: the data for training
        dims: the dimension for your network architecture
        L2Reg: the weight for l2 norm
        SparsityReg: the weight for sparsity
        SparsityPro': the target level sparsity 
        MaxEpochs: maximum training epochs 
        Versbose: tracking the loss or ont 
    '''
    ## Prepare for the training 
    # set the hyper-parameter 
    HyperParams = DefaultParams
    for key in kwargs.keys():
        HyperParams[key] = kwargs[key]
    # init the model
    model = AE( dims)
    # decide optimizer 
    optimizer = Adam( model.parameters(), lr=1e-3, 
                        weight_decay=HyperParams['L2Reg'])
    # some storages
    losses  = [] 

    ## Start training
    for epoch in range(HyperParams['MaxEpochs']):
        #tqdm(range(HyperParams['MaxEpochs'])): 
        # reshape the image
        x = torch.FloatTensor( data).view( data.shape[0], -1)
        # reconstruct x
        z, x_hat =  model( x)
        # calculate the loss 
        loss = MSEsparse( x, x_hat, z, 
                            spa_reg=HyperParams['SparsityReg'], 
                            p=HyperParams['SparsityPro'])
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # store the te losses for plotting 
        losses.append(loss)
        # track training
        if (epoch%10 ==0) and HyperParams['Verbose']:
            print( f'Epoch:{epoch}, Loss:{loss}')
        # decide if end
        
    return model 
    
