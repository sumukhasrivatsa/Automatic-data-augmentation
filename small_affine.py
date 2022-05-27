
import torch.nn.functional as F
from torch.distributions import Uniform
import torch.nn as nn
import torch

###for netA
class Affine(nn.Module):
  def __init__(self,image,label):
    super(Affine, self).__init__()
    self.nz=6
    self.image=image
    self.label=label
    
    self.mean = torch.tensor(0)
    self.std = torch.tensor(0)


    ########### ACTUAL NETA
    self.fc_loc = nn.Sequential(
            nn.Linear(self.nz, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 6  ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    self.fc1=nn.Linear(6,4)


  def get_affine_matrix(self,noise):

     identitymatrix = torch.eye(2, 3)
     #print(identitymatrix)
     identitymatrix = identitymatrix.unsqueeze(0)
     #print(identitymatrix)
     identitymatrix = identitymatrix.repeat(noise.shape[0], 1, 1)
     #print(identitymatrix)
     theta = self.fc_loc(noise)
     theta = self.fc1(theta)


     affinematrix = identitymatrix
     affinematrix[:, 0, 0] = theta[:, 0]
     affinematrix[:, 0, 1] = theta[:, 1]
     affinematrix[:, 1, 0] = theta[:, 2]
     affinematrix[:, 1, 1] = theta[:, 3]
     
     return affinematrix


  def forward(self):
    if self.mean.device != self.image.device:
            self.mean = self.mean.to(self.image.device)
            self.std = self.std.to(self.image.device)
    bs = self.image.shape[0]
    self.uniform = Uniform(low=-torch.ones(bs, self.nz), high=torch.ones(bs, self.nz))
    #print(self.uniform)
    noise = self.uniform.rsample()
    #print(noise)
    # get transformation matrix
    
    affinematrix = self.get_affine_matrix(noise)

  
    # compute transformation grid
    grids = F.affine_grid(affinematrix, self.image.size(), align_corners=True)
    
    # apply transformation
    x = F.grid_sample(self.image, grids, align_corners=True)
    
    
    return x,self.label
    

    
        
                
