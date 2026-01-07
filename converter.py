import numpy as np
import torch

#Abstract class for converting between environment data and agent compatible features
class FeatureMeta():
    '''Abstract class for converting between environment data and agent compatible features'''
    def __init__(self, name, position):
        '''Constructor containing the position of the feature in environment and its name in the agent'''
        self.name = name
        self.position = position
    def input_process(self, x):
        '''Convert environment data to agent compatible features'''
        pass
    def output_process(self, x):
        '''Convert agent compatible features to environment data'''
        pass
    def get_feature_length(self):
        '''Get the length of the features for the agent'''
        return 1

class ContinousFeatureMeta(FeatureMeta):
    '''A feature converter for continous data'''
    def __init__(self,name, position, min, max):
        '''Constructor, set the name for agent and the position in the environment data, and define the range of the data for normalisation'''
        super().__init__(name,position)
        self.min = min
        self.max = max
    def input_process(self, x):
        '''Normalise environment data into agent feature'''
        return (x - self.min) / (self.max - self.min)
    def output_process(self, x):
        '''Denormalise agent feature into environment data'''
        return x * (self.max - self.min) + self.min
class ClassificationFeatureMeta(FeatureMeta):
    '''A feature converter for classification data'''
    def __init__(self,name,position,values : list):
        '''Constructor defining the name of the feature set and position in environment data, as well as the possible values of the feature'''
        super().__init__(name,position)
        #Create a mapping from possible values of environment data to a unary encoding for the Agent
        self.pos_to_value = {}
        self.value_to_pos = {}
        self.length = len(values)
        i = 0
        for value in values:
            self.pos_to_value[i] = value
            self.value_to_pos[value] = i
            i+=1
    def input_process(self, x):
        '''Map classification data from the environment into a unary encoding for the agent'''
        #features = np.array(dtype=float).zeros(self.length)
        features = np.zeros(self.length)
        features[self.value_to_pos[x]] = 1
        return features
    def output_process(self, x):
        '''Map unary encoding from the agent into classification data for the environment'''
        identified_positions = np.where(x == 1)[0]
        values = []
        for pos in identified_positions:
            values.append(self.pos_to_value[pos])
        return values
    def get_feature_length(self):
        '''Get the number of features used to encode the classification data in unary format'''
        return self.length


class Converter():
    '''A agent component that converts between environment data and agent compatible features'''
    def __init__(self,data_info: list[dict]):
        '''Constructor, takes a specification of different environmental inputs (data_info) and generates a encoding schema for the agent along with the required converters'''
        #Generate the converters
        self.input_metas = self.get_position_metas(data_info)
        #Calculate the expected length of inputs from the environment
        self.denormalised_data_length = len(data_info)
        #Calculate the number of features output by the converters
        self.normalised_feature_length = 0
        for meta in self.input_metas.values():
            self.normalised_feature_length += meta.get_feature_length()

    def get_position_metas(self, meta_list: list[dict]):
        '''Generate the converters for each piece of environment data based on environment specifications'''
        metas = {}
        for feature in meta_list:
            name = feature["name"]
            position = feature["position"]
            #Make a continous converter for continous data and a classification converter for classification data
            match feature["type"]:
                case "continous":
                    min = feature["min"]
                    max = feature["max"]
                    metas[position] = ContinousFeatureMeta(name,position,min,max)
                case "classification":
                    values = feature["values"]
                    metas[position] = ClassificationFeatureMeta(name,position,values)
        return metas

    def normalise(self, inputs : np.ndarray) -> torch.Tensor:
        '''Take environment data and convert it into features for the agent'''
        col_index = len(inputs.shape) - 1
        #If the shape of the environment data doesn't meet what the converter was built for then we can't continue
        if inputs.shape[col_index] != self.denormalised_data_length:
            raise ValueError("input dimensions don't match specifications")
        
        #Generate the features by doing a feature wise normalisation using the converter components
        #normalised_inputs = np.array(dtype=float).zeros(self.normalised_feature_length)
        normalised_inputs = np.zeros(self.normalised_feature_length)
        i = 0
        for meta in self.input_metas.values():
            normalised_inputs[i:i+meta.get_feature_length()] = meta.input_process(inputs[meta.position])
            i += meta.get_feature_length()
        #Return the features as a torch tensor
        return torch.from_numpy(normalised_inputs).float()

    def denormalise(self, pre_outputs: torch.Tensor) -> np.ndarray:
        '''Take agent features and convert them into environment data'''
        col_index = len(pre_outputs.shape) - 1
        #If the shape of the agent data doesn't meet what the converter was built for then we can't continue
        if pre_outputs.shape[col_index] != self.normalised_feature_length:
            raise ValueError("output dimensions don't match specifications")
        
        #Generate the environment data by doing a feature wise denormalisation using the converter components
        #denormalised_outputs = np.array(dtype=float).zeros(self.normalised_feature_length)
        denormalised_outputs = np.zeros(self.normalised_feature_length)
        numpy_pre_outputs = pre_outputs.detach().numpy()
        i = 0
        for meta in self.input_metas.values():
            denormalised_outputs[i:i+meta.get_feature_length()] = meta.output_process(numpy_pre_outputs[meta.position])
            i += meta.get_feature_length()
        #Return the environment data as a numpy ndarray
        return denormalised_outputs
    def get_input_length(self):
        '''Get the length of the environment data accepted by the converter'''
        return self.input_length
    def get_feature_length(self):
        '''Get the length of the feature set output by the converter'''
        return self.denormalised_data_length