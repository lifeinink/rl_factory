import torch

# Simple placeholder class to hold torch compatible features for agents that can later be replaced with more sophisticated replay buffers
class ReplayBuffer:
    '''Simple placeholder class to hold torch compatible features for agents that can later be replaced with more sophisticated replay buffers'''

    #Define each of the feature clusters in the replay buffer
    #past_observation, action, observation, reward, done
    def __init__(self,capacity=-1):
        '''Define each of the feature clusters in the replay buffer'''
        self.capacity = capacity
        self.past_obs_buffer = None
        self.action_buffer = None
        self.obs_buffer = None
        self.reward_buffer = None
        self.done_buffer = None
    
    #Add another row to the replay buffer
    def add(self,past_obs : torch.Tensor,action : torch.Tensor,obs : torch.Tensor,reward : torch.Tensor,done : torch.Tensor):
        '''Add another row to the replay buffer'''
        #If the replay buffers are empty, initialise them
        if self.past_obs_buffer is None:
            self.past_obs_buffer = past_obs
            self.action_buffer = action
            self.obs_buffer = obs
            self.reward_buffer = reward
            self.done_buffer = done
        #If the replay buffers are not empty, append them
        else:
            self.past_obs_buffer = torch.cat((self.past_obs_buffer,past_obs),dim=0)
            self.action_buffer = torch.cat((self.action_buffer,action),dim=0)
            self.obs_buffer = torch.cat((self.obs_buffer,obs),dim=0)
            self.reward_buffer = torch.cat((self.reward_buffer,reward),dim=0)
            self.done_buffer = torch.cat((self.done_buffer,done),dim=0)