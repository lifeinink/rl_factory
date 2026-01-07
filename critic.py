import batcher
import torch

#Abstract class for critic components of actor-critic RL agents
class Critic:
    '''Abstract class for critic components of actor-critic RL agents'''
    def __init__(self,models):
        self.models = models
    
    #def prime_update(self,past_observations: torch.Tensor,result_observations: torch.Tensor,result_rewards: torch.Tensor,predicted_actions: torch.Tensor, is_done: bool=False):
    #    pass
    def get_actor_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        '''Calculates the loss for the actor network.'''
        pass
    
    #def critic_update(self):
    #    pass
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, next_actions: torch.Tensor):
        """Updates both critic networks based on a batch of experience."""
    
    def target_update(self):
        '''Perform a soft update on the target networks'''
        pass