import torch
import numpy as np
import converter
import replay_buffer

class Model:
    '''An abstract class for the model components of an RL agent produced by the factory with the common methods used by its agents'''
    def __init__(self,observation_converter : converter.Converter,action_converter : converter.Converter, buffer_size=100000):
        '''Constructor, creates a basic replay buffer and saves the converters provided'''
        self.steps = 0
        self.steps_since_update = 0
        #self.replay_buffer = torch.Tensor((buffer_size, 5),dtype=torch.Tensor)
        self.replay_buffer = replay_buffer.ReplayBuffer(buffer_size)
        self.observation_converter = observation_converter
        self.action_converter = action_converter

    def step(self, observation):
        '''Get an action from the model, track number of steps and steps since last update'''
        self.steps += 1
        self.steps_since_update += 1
        return None # this is where the model would choose an action
    
    def normalise_observations(self,observations : np.ndarray):
        '''Convert environment observations into agent features'''
        return self.observation_converter.normalise(observations)
    
    #Extra manipulation because actions can be 1 to -1
    def denormalise_actions(self,actions : torch.Tensor) -> np.ndarray:
        '''Convert agent features into environment actions'''
        return self.action_converter.denormalise((actions+1)/2.0)
    
    def normalise_actions(self,actions : np.ndarray) -> torch.Tensor:
        '''Convert environment actions into agent features'''
        return self.action_converter.normalise(actions)

    def learn(self, past_observation, action, observation, reward, done=False):
        '''Adds an example to the replay buffer with modifications for the action range'''
        done = 1 if done else 0
        past_observation = self.normalise_observations(past_observation).unsqueeze(0)
        observation = self.normalise_observations(observation).unsqueeze(0)
        #actor range is 1 to -1 for now so manually adjusting
        action = (self.normalise_actions(action) * 2.0 - 1.0).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)
        #append to replay buffer
        #self.replay_buffer= np.cat((self.replay_buffer, torch.tensor([past_observation, action, observation, reward, done])), dim=0)
        self.replay_buffer.add(past_observation, action, observation, reward, done)
        # this is where the model would learn from the observation and reward