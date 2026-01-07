import batcher
import torch
import torch.nn as nn
import copy

class Actor:
    '''Actor component of an actor critic agent'''
    def __init__(self,model, tau=0.001):
        '''Constructor, sets the model, its respective target model, and the soft update parameter'''
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.tau = tau
        self.is_odd_update = False
    
    def step(self,observation: torch.Tensor) -> torch.Tensor:
        '''Perform a inference step to get actions'''
        return self.model.forward(observation)
    
    def target_step(self,observation : torch.Tensor) -> torch.Tensor:
        '''Perform a inference step to get actions with the target model'''
        return self.target_model.forward(observation)
    
    def batch_step(self,observations : torch.Tensor) -> torch.Tensor:
        '''Perform a inference step to get actions for a batch of observations'''
        return self.model.forward(observations)
    
    def update(self,loss: torch.Tensor):
        '''Update model weights directly with loss, and soft update target model if necessary'''
        #Update model
        self.model.loss_direct_train(loss)
        #Every other update update the target model
        if self.is_odd_update:
            self.target_model.soft_update(self.model,self.tau)
        #Update the target model update flag
        self.is_odd_update = not self.is_odd_update

    
    def soft_update(self,target):
        '''Perform a soft update on the target model'''
        self.target_model.soft_update(target,self.tau)