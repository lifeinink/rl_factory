import critic
import numpy as np
import torch
import copy

# A duelling double critic: takes the lowest reward predicted from two double critics, target critics are soft updated from leading critics
class DuellingDoubleCritic(critic.Critic):
    '''A duelling double critic: takes the lowest reward predicted from two double critics, target critics are soft updated from leading critics'''
    #Constructor
    def __init__(self,critic_one,critic_two, tau=0.001, discount_factor=0.8):
        #Set the two critics and make their target networks
        self.critic_one = critic_one
        self.critic_two = critic_two
        self.target_one = copy.deepcopy(critic_one)
        self.target_two = copy.deepcopy(critic_two)
        #Set hyperparameters according to arguments (tau for how aggresive the soft updates are, discount factor for degree of taking into account the future)
        self.tau = tau
        self.discount_factor = discount_factor
        #Set models by calling the parent class constructor
        models = [critic_one,critic_two,self.target_one,self.target_two]
        super().__init__(models)
    
    
    def get_actor_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Calculates the loss for the actor network."""
        # The actor loss is the negative mean of the Q-value from the first critic.
        # This encourages the actor to output actions that maximize the Q-value.
        q_values_one = self.critic_one(torch.cat([states, actions], 1))
        q_values_two = self.critic_two(torch.cat([states, actions], 1))
        q_values = torch.min(q_values_one, q_values_two)
        return -q_values.mean()
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, next_actions: torch.Tensor):
        """Updates both critic networks based on a batch of experience."""
        
        # Concatenate states and actions for model input
        states_actions = torch.cat([states, actions], 1)

        # Compute the target Q-value
        with torch.no_grad():
            # Get the minimum Q-value from the two target critics for the next state
            next_q1 = self.target_one(torch.cat([next_states, next_actions], 1))
            next_q2 = self.target_two(torch.cat([next_states, next_actions], 1))
            min_next_q = torch.min(next_q1, next_q2)
            
            # The target Q-value (y)
            target_q = rewards + self.discount_factor * (1 - dones) * min_next_q

        # Train both critic networks
        self.critic_one.train(states_actions, target_q)
        self.critic_two.train(states_actions, target_q)
    
    #Soft update the target networks according to the parameters of the main critics
    def target_update(self):
        '''Soft update the target networks according to the parameters of the main critics'''
        self.target_one.soft_update(self.critic_one,self.tau)
        self.target_two.soft_update(self.critic_two,self.tau)