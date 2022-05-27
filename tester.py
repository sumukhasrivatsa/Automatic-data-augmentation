bs=5
from numpy import float32
import torch
from torch import cos_
from torch.distributions import Uniform

uniform=Uniform(low=-torch.ones(bs, 6), high=torch.ones(bs, 6))
noise = uniform.rsample()
print(noise)
print(noise.shape)


identitymatrix = torch.eye(2, 2)
#print(identitymatrix)
identitymatrix = identitymatrix.unsqueeze(0)
#print(identitymatrix)
identitymatrix = identitymatrix.repeat(noise.shape[0], 1, 1)
print(identitymatrix)
theta=torch.tensor(1.0471975512)
theta = theta.to(torch.float32)
affinematrix = identitymatrix
affinematrix[:, 0, 0] = torch.cos(theta)
affinematrix[:, 0, 1] = -torch.sin(theta)
affinematrix[:, 1, 0] = torch.sin(theta)
affinematrix[:, 1, 1] = torch.cos(theta)


print(affinematrix)