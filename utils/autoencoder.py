import torch
import torch.nn as nn 
from torch.optim import Adam

#--------------------------------
#        Hyperparameters
#--------------------------------

DefaultParams = { 
                'LR': 1e-4,
                'MaxEpochs': 15,
                'L2Reg': 1e-5,
                'SparsityReg': 5,
                'SparsityPro': .1,
                'Verbose': True,
                'BatchSize': 32,
                'If_gpu': False, 
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
        Reference
        https://www.mathworks.com/help/deeplearning/ref/trainautoencoder.html
        
            rho_hat = 1/M Sum_m sigmoid( W.T @ x + b) 
                m is number of input data 
        '''
        # get the target tesnor 
        rho_hat = z.mean(dim=0)
        rho     = p_tar#(torch.ones_like( rho_hat) * p_tar).to(self.device)

        # calculate kld 
        kld1 = rho * (rho/(rho_hat+eps_)+eps_).log()
        kld2 = (1-rho) * ((1-rho)/(1-rho_hat+eps_)+eps_).log()
        return (kld1 + kld2).sum()

#---------------------------
#    Train Auto encoder 
#---------------------------

def trainAE( train_data, model, **kwargs):
    '''Train a sparse autoencoder

    Input:
        data: the data for training
        model: the AE model for training 
        LR: learning rate 
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
    n_batch = int( len(x) / HyperParams['BatchSize'])
    x_tensor = x.type( torch.FloatTensor)
    y_tensor = y.type( torch.FloatTensor)
    _dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    _dataloader = torch.utils.data.DataLoader( _dataset, 
                batch_size=HyperParams['BatchSize'], drop_last=True)
    # decide optimizer 
    optimizer = Adam( model.parameters(), lr=HyperParams['LR'], 
                        weight_decay=HyperParams['L2Reg'])       
    ## get batch_size
    losses = []
    
    # start training
    model.train()
    for epoch in range( HyperParams['MaxEpochs']):

        ## train each batch 
        loss_ = 0        
        for _, (x_batch, _) in enumerate(_dataloader):

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
            print( f'   Epoch:{epoch}, Loss:{loss_}')

    return model, losses 