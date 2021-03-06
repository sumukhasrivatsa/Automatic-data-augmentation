import torch.nn.functional as F
from torch.distributions import Uniform
import torch.nn as nn
import torch


class netC(nn.Module):
    def __init__(self):
        super(netC, self).__init__()
        
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        #(10,3,350,350)
       
        #Input shape= (10,1,350,350)
        
        self.conv1=nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=1,padding=1)
        #Shape= (32,2,28,28)
        self.bn1=nn.BatchNorm2d(num_features=2)
        #Shape= (32,2,28,28)
        self.relu1=nn.ReLU()
        #Shape= (32,2,28,28)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (32,2,14,14)
        
        
        self.conv2=nn.Conv2d(in_channels=2,out_channels=5,kernel_size=3,stride=1,padding=1)
        #Shape= (32,5,14 14)
        self.relu2=nn.ReLU()
        #Shape= (32,5,14 14)
        self.fc1=nn.Linear(in_features=5*14*14,out_features=100)
        self.fc2=nn.Linear(100,10)
        self.out_=nn.Softmax()
    
    def forward(self,images,labels):
        inputs=images
        output=self.conv1(inputs)
        #print(output.shape)
        output=self.bn1(output)
        output=self.relu1(output)
        #print(output.shape)
            
        output=self.pool(output)
        #print(output.shape)    
        output=self.conv2(output)
        #print(output.shape)
        output=self.relu2(output)
        #print(output.shape)
            
        
            
            
        #Above output will be in matrix form, with shape (10,2,25,25)
            
        output=output.view(-1,5*14*14)
            
            
        output=self.fc1(output)
        #print(output.shape)
        output=self.fc2(output)
        #print(output.shape)
        output=self.out_(output)
        #print(output.shape)
            
        return output


        