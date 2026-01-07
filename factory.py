import numpy as np
import converter
import ffnnet_model
import anfis_model
import actor as ac
import duelling_double_critic
import actor_critic
#import batcher

# Factory for making RL actor critic agents
class ModelFactory():
    '''Factory for making RL actor critic agents'''
    def __init__(self):
        '''Constructor, initialises agent specifications'''
        #Initialise agent specifications
        self.models = {}
        self.observation_converter = None
        self.action_converter = None

    #Creates the mapping between the environment variables and the features and output of the agent
    def set_converter(self, input_meta: list[dict], output_data: list[dict]):
        '''Creates the mapping between the environment variables and the features and output of the agent'''
        self.observation_converter = converter.Converter(input_meta)
        self.action_converter = converter.Converter(output_data)
        return self
    
    # Create a simple DNN model, role specifies whether it's loaded into the actor or critic component
    def predictor_DNN(self,role="actor"):
        '''Create a simple DNN model, role specifies whether it's loaded into the actor or critic component'''
        #The model can't be made if we don't know what the inputs and outputs are so a converter must be defined first
        if self.observation_converter == None or self.action_converter == None:
            raise ValueError("converter not set")
        #Get the number of features and outputs
        input_length = self.observation_converter.get_feature_length()
        output_length = self.action_converter.get_feature_length()
        #If the model is for a critic it needs to take into account rewards in the previous step. Pretty sure this is a mistake but TODO: want to test to make sure later
        if role == "critic":
            input_length += output_length
            output_length = 1
        # Create the hidden layer node numbers based on the 2/3 of previous layer lengths heuristic
        hidden_layers = []
        current_layers = 2.0/3.0*input_length+output_length
        while current_layers > output_length:
            hidden_layers.append(int(current_layers))
            current_layers = 2.0/3.0*current_layers
        hidden_layers.append(output_length)
        #Create the model and save it for agent assembly
        self.models[role] = ffnnet_model.DenseFFNN(input_length,hidden_layers)
        return self
    # Create a dense FNN as a critic according to rough heuristics for RL critics
    def RL_DQ(self,role="critic",hidden_layers_num = 2, binary_multiple = 8, double=False):
        '''Create a dense FNN as a critic according to rough heuristics for RL critics'''
        #The model can't be made if we don't know what the inputs and outputs are so a converter must be defined first
        if self.observation_converter == None or self.action_converter == None:
            raise ValueError("converter not set")
        
        # Define the number of features based on the RL agents inputs an outputs assuming it's a critic
        input_length = self.observation_converter.get_feature_length() + self.action_converter.get_feature_length()
        output_length = 1
        #But if it's actually an actor then correct the number of features
        if role == "actor":
            input_length = self.observation_converter.get_feature_length()
            output_length = self.action_converter.get_feature_length()

        #Create node per layer specifications as some multiple of two for the number of hidden layers (assumes this is the first iteration of the agent, does not finetune based on previous results)
        hidden_layers = []
        current_layers = 2^binary_multiple
        for i in range(hidden_layers_num):
            hidden_layers.append(int(current_layers))
            current_layers = 2^binary_multiple
            binary_multiple -= 1
        hidden_layers.append(output_length)
        #Made the model according to the specifications
        model = ffnnet_model.DenseFFNN(input_length,hidden_layers)
        #Save the model for agent assembly
        if not role in self.models:
            self.models[role] = []
        self.models[role].append(model)
        if double:
            self.RL_DQ(role,hidden_layers_num,binary_multiple,False)
        return self

    # Creates a neuro-fuzzy model for use as a component of the agent, by default as an actor component
    def ANFIS(self,role="actor"):
        '''Creates a neuro-fuzzy model for use as a component of the agent, by default as an actor component'''
        #The model can't be made if we don't know what the inputs and outputs are so a converter must be defined first
        if self.observation_converter == None or self.action_converter == None:
            raise ValueError("converter not set")
        #Determine the number of features and outputs assuming it's an actor
        input_length = self.observation_converter.get_feature_length()
        output_length = self.action_converter.get_feature_length()
        #If it's actually a critic correct that. Prettry sure this is a mistake.
        if role == "critic":
            input_length += output_length
            output_length = 1
        
        #Assume the model's number of rules is approximately that of a fuzzy associative memory matrix, construct the model and save it for the later assembly of the agent
        rule_num = input_length ^ 2 * output_length
        self.models[role] = anfis_model.XanFISModel(input_length,output_length,rule_num)
        return self
    
    #This used to make a sample batcher, but is no longer needed since the batcher is no longer used
    #def set_batcher(self,batch_size):
    #    self.batcher = batcher.Batcher(batch_size)
    #    return self


    #Create a TD3 agent from the earlier defined components (requires an actor model component, a critic model component and converters)
    def TDDD_build(self):
        '''Create a TD3 agent from the earlier defined components (requires an actor model component, a critic model component and converters)'''
        #Batcher no longer needed
        #if self.batcher == None:
        #    raise ValueError("batcher not set")

        #Ensure all components are ready
        if "actor" not in self.models or "critic" not in self.models:
            raise ValueError("models not set")
        if len(self.models["critic"]) != 2:
            raise ValueError("critic models not set")
        
        #Make the actor and critic components
        actor = ac.Actor(self.models["actor"])
        critic = duelling_double_critic.DuellingDoubleCritic(self.models["critic"][0],self.models["critic"][1])
        #Assemble the actor critic from the actor and critic components and return it
        return actor_critic.ActorCritic(actor,critic,self.observation_converter,self.action_converter)