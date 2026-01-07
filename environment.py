import gymnasium as gym
import factory
from control_model import Model
import numpy as np
from symbolic_model import FuzzyModel

#Creates a reward that can be hill climbed to balancing the cartpole in the center via the harmonic mean of the normalised displacement from the center and the pole being upright
def make_shaped_reward(reward,observation,terminated,truncated):
    '''Creates a reward that can be hill climbed to balancing the cartpole in the center via the harmonic mean of the normalised displacement from the center and the pole being upright'''
    position = 1 - abs(observation[0])
    angle = (0.2 - abs(observation[1]))/0.2
    if terminated: return -1
    return position * angle / (position + angle)

#Test loop for MuJoCo based Cartpole 
#TODO: adjust RL results to 250 iteration?
def run_cartpole(model : Model,max_iterations=250):
    '''Test loop for MuJoCo based Cartpole'''
    #Initialise the environment
    env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1, render_mode="rgb_array")
    observation, info = env.reset(seed=42)
    rewards = np.zeros(max_iterations)

    # Run the simulation for the number of iterations specified
    for i in range(max_iterations):
        #print(observation)

        #Get the agent's actions and calculate the next timepoint
        action = model.step(observation)
        past_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        #Shape the reward to reduce sparsity or a non parametric hill
        reward = make_shaped_reward(reward,observation,terminated,truncated)
        rewards[i] = reward
        #Add the relevant features to the replay buffer (and update the model if it's an online one)
        model.learn(past_observation,action,observation, reward,terminated)
        if terminated or truncated:
            observation, info = env.reset()
    
    #Clean up and return the rewards per step array
    env.close()
    return rewards

#Just plots reward series
def print_reward_curve(rewards: np.ndarray):
    '''Just plots reward series'''
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.show()

#Takes a path to a fll file and tests it against MuJoCo CartPole
def fuzzy_test(fuzzy_spec_dir):
    '''Takes a path to a fll file and tests it against MuJoCo CartPole'''
    #Read the fuzzy system specification
    spec_str = ""
    with open(fuzzy_spec_dir,"r") as f:
        spec_str = f.read()

    #Make a fuzzy controller
    agent = FuzzyModel(spec_str,["position","angle","velocity","angular_velocity"],["force"],1)
    #Run the simulation
    rewards = run_cartpole(agent)
    # Save the "replay buffer" and return the cumulative reward
    agent.save_memory()
    print(rewards.sum())
    return rewards.sum()

def run_RL():
    '''Run the RL agent and collect cumulative reward data'''
    #Specify the inputs and outputs
    inputs = []
    inputs.append({"name":"position","position":0,"type":"continous","min":-1,"max":1})
    inputs.append({"name":"angle","position":1,"type":"continous","min":-0.2,"max":0.2})
    inputs.append({"name":"velocity","position":2,"type":"continous","min":-2.0,"max":2.0})
    inputs.append({"name":"angular_velocity","position":3,"type":"continous","min":-2.0,"max":2.0})
    outputs = []
    outputs.append({"name":"action","position":0,"type":"continous","min":-3,"max":3})

    #Make a RL agent, either a ANFIS based TD3 or a normal TD3
    ac_agent = factory.ModelFactory().set_converter(inputs,outputs).ANFIS().RL_DQ(double=True).TDDD_build()
    #ac_agent = factory.ModelFactory().set_converter(inputs,outputs).predictor_DNN().RL_DQ(double=True).TDDD_build()

    #Train the RL agent over 100 iterations and save the culumative rewards per simulation
    iteration_rewards = []
    for i in range(0,100):
        iteration_rewards.append(run_cartpole(ac_agent).sum())
        with open("rl_reward_curve.txt","w+") as f:
            f.write(str(iteration_rewards))
        print(iteration_rewards[i])
        ac_agent.update()
    print_reward_curve(np.array(iteration_rewards))

if __name__ == "__main__":
    #fuzzy_test("fl_specs/claude_v4.fl")
    run_RL()