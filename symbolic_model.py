import fuzzylite as fl
import numpy as np
import pandas as pd

# A placeholder model that runs a fuzzy logic system
class FuzzyModel():
    # Constructor, takes a mapping from inputs by position to their corrisponding inputs in the fuzzy system and the outputs from the fuzzy system to their corrisponding actions, and the path to the fuzzylite specification for the fuzzy system
    def __init__(self, fl_specs,obs_titles: list[str],action_order: list[str],num_rewards: int):
        '''onstructor, takes a mapping from inputs by position to their corrisponding inputs in the fuzzy system and the outputs from the fuzzy system to their corrisponding actions, and the path to the fuzzylite specification for the fuzzy system'''
        self.specs = fl_specs
        #Save the mappings from the environment to the fls
        self.obs_titles = obs_titles
        self.action_order = action_order
        #Load the fuzzy logic engine
        self.load_control_sys()

        #Initialise "replay_buffer"
        obs_cols = obs_titles
        past_obs_cols = []
        for i in range(len(obs_titles)):
            past_obs_cols.append("past_"+obs_titles[i])
        reward_cols = []
        for i in range(num_rewards):
            reward_cols.append("reward_"+str(i))
        action_cols = action_order
        cols = obs_cols + past_obs_cols + reward_cols + action_cols
        self.memory = pd.DataFrame(columns=cols)
    
    #Loads the fuzzylite specifications into a fuzzylite engine
    def load_control_sys(self):
        '''Loads the fuzzylite specifications into a fuzzylite engine'''
        try:
            self.ctrl_sys : fl.Engine = fl.FllImporter().from_string(self.specs)
        except Exception as e:
            print(e.with_traceback())
            raise Exception("Error loading control system")
    
    # Takes inputs, maps them to the predefined input terms, runs them through the fls, then maps to outputs the environment can use
    def step(self,observation: np.ndarray):
        '''Takes inputs, maps them to the predefined input terms, runs them through the fls, then maps to outputs the environment can use'''
        #Briefly check to make sure the superficial match between environment inputs and those of the fls match
        if len(observation) != len(self.obs_titles):
            raise ValueError("Spec observation mismatch")
        
        #Set the input variables for the fls
        for i in range(len(observation)):
            self.ctrl_sys.input_variable(self.obs_titles[i]).value = observation[i]

        # Run the fls
        self.ctrl_sys.process()

        #Map output variables to their respective outputs for the environment and return them
        action = np.zeros(len(self.action_order))
        for i in range(len(self.action_order)):
            action[i] = self.ctrl_sys.output_variable(self.action_order[i]).value
        return action
    
    # Adds relevant data for analysis to the replay buffer. Useful for heuristic optimisation.
    def learn(self, past_observation, action, observation, reward, done=False):
        '''Adds relevant data for analysis to the replay buffer. Useful for heuristic optimisation.'''
        new_row = pd.Series()
        for i in range(len(self.obs_titles)):
            new_row[self.obs_titles[i]] = observation[i]
            new_row["past_"+self.obs_titles[i]] = past_observation[i]
        for i in range(len(self.action_order)):
            new_row[self.action_order[i]] = action[i]
        if isinstance(reward,np.float64) or isinstance(reward,float) or isinstance(reward,int):
            reward = [reward]
        for i in range(len(reward)):
            new_row["reward_"+str(i)] = reward[i]
        self.memory = pd.concat([self.memory,new_row.to_frame().T],ignore_index=True)
    
    #Saves the "replay buffer" to a csv file for later use
    def save_memory(self):
        '''Saves the "replay buffer" to a csv file for later use'''
        self.memory.to_csv("memory.csv")
