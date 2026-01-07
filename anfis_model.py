from xanfis import GdAnfisRegressor
import numpy as np
import torch
import copy

class XanFISModel():
    '''A wrapper for the xanfis model that is compatible as a model component for the RL factory'''
    def __init__(self, num_inputs: int, num_outputs: int, num_rules: int):
        '''Initialise the xanfis model according to the number of inputs, outputs and rules specified, and do a training cycle on random data to enable inference'''
        self.model : GdAnfisRegressor = GdAnfisRegressor(num_rules=num_rules, vanishing_strategy="blend", mf_class="Gaussian", optim="Adam", batch_size=1, verbose=True,seed=42)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_rules = num_rules
        #Train xanfis model on random data so that it can do inference
        self.random_init_fit()

    def random_init_fit(self):
        '''Creates a dummy set of features with no relationship to the targets to fit the model to so it can do inference and start the model with good enough initial weights'''
        #Make the random data
        x = np.random.rand(100, self.num_inputs)
        y = np.random.rand(100, self.num_outputs) * 2 - 1
        #Fit the random data to the model
        self.model.fit(x, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Perform a inference step with the xanfis model and return the output'''
        #If the inputs are a lump, extract the features before performing inference
        if len(x.shape) == 1:
            return self.model.network(x.unsqueeze(0)).squeeze(0)
        # If the inputs don't match the shape specified something has gone wrong with the setup
        if self.num_inputs != x.shape[1]:
            raise ValueError("Input shape does not match model input shape")
        #Perform the inference and return the output
        return self.model.network(x)

    def train(self, x: torch.Tensor, y: torch.Tensor) -> float:
        '''Train the xanfis model on labelled data and return the loss'''
        #If the inputs and outputs to be trained on don't match what the model was made for then something has gone wrong with the setup
        if self.num_inputs != x.shape[1] or self.num_outputs != y.shape[1]:
            raise ValueError("Input and output shapes do not match model input and output shapes")
        
        # Convert tensors to numpy for xanfis
        x_np = x.detach().numpy()
        y_np = y.detach().numpy()

        #Train the model and return the loss
        self.model.fit(x_np, y_np)
        return self.model.loss

    def soft_update(self, leader_model, tau: float):
        '''Make a soft update to this model according to a leader model with aggresiveness specified by tau'''
        #Get the model weights
        self_parameters = self.model.network.parameters()
        leader_parameters = leader_model.model.network.parameters()
        #Update the weights
        for self_param, leader_param in zip(self_parameters, leader_parameters):
            self_param.data.copy_(tau * leader_param.data + (1 - tau) * self_param.data)
            
    def loss_direct_train(self, loss: torch.Tensor):
        '''Train the xanfis model without regularisation directly on loss'''

        #DEBUG: store current weights so you can check later if they've actually updated
        #curr_weights = copy.deepcopy(self.model.network.state_dict())

        #Set the mode of the model
        self.model.network.train() 
        
        #Perform back propogation
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        #DEBUG: check if the weights have actually updated
        #new_weights = self.model.network.state_dict()
        #if str(curr_weights) == str(new_weights):
        #    raise Exception("It didn't update")

        #Return the loss
        return loss.item()
        
