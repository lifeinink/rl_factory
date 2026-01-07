import numpy as np
import torch.nn as nn
import torch


# A simple FFNN for use as a component of DRLs with hardcoded activation functions
class DenseFFNN(nn.Module):
    '''A simple FFNN for use as a component of DRLs with hardcoded activation functions'''
    #Constructor, to initialise the NNet structure
    def __init__(self,input_num: int,layers: list[int], init_learning_rate: float=0.001):
        '''Constructor, to initialise the NNet structure'''
        super().__init__()
        # Initialise the nnet layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_num,layers[0]))
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i],layers[i+1]))
        
        #Define the loss function and optimiser (mean squared error + adam optimiser starting at the specified learning rate)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=init_learning_rate)
    
    #Inference step
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        '''Inference step'''
        # Apply activation to all but the last layer
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        # No activation on the output layer
        return self.layers[-1](x)
    
    # Perform a single back propogation on labelled data
    def train(self,inputs : torch.Tensor,targets : torch.Tensor) -> float:
        '''Perform a single back propogation on labelled data'''
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    # Perform a single back propogation with predefined losses
    def loss_direct_train(self,loss: torch.Tensor) -> float:
        '''Perform a single back propogation with predefined losses'''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    #Perform a soft update to this NNet based on a leader NNet presumed to have the same architecture
    def soft_update(self,leader_model: nn.Module,tau):
        '''Perform a soft update to this NNet based on a leader NNet presumed to have the same architecture'''
        for self_param, leader_param in zip(self.parameters(), leader_model.parameters()):
            self_param.data.copy_(tau * leader_param.data + (1 - tau) * self_param.data)