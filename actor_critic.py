from control_model import Model
import numpy as np
import actor
import critic
import converter
import torch
class ActorCritic(Model):
    '''An actor critic agent that can be constructed by the RL factory'''
    def __init__(self, actor : actor.Actor, critic : critic.Critic, observation_converter: converter.Converter, action_converter: converter.Converter,action_noise=0.01,buffer_size=100000,actor_update_delay=2):
        '''Constructor, sets the converters, the actor and critic models as well as the desired amount of noise and how often to update the actor'''
        super().__init__(observation_converter, action_converter,buffer_size)
        self.actor = actor
        self.critic = critic
        self.action_noise = action_noise
        self.actor_update_delay = actor_update_delay
        self.number_of_updates = 0

    def step(self, observation : np.ndarray):
        '''Perform a inference step to get actions'''
        super().step(observation)
        #Normalise the features
        normalised_observation = super().normalise_observations(observation)
        #Perform inference to get the actions
        normalised_actions = self.actor.step(normalised_observation)
        #Denormalise the actions
        action = super().denormalise_actions(normalised_actions)
        #Return the actions
        return action

    def update(self):
        '''Update the actor and critic models'''
        #Retrieve the replay buffer
        rewards = self.replay_buffer.reward_buffer
        future_states = self.replay_buffer.obs_buffer
        initial_states = self.replay_buffer.past_obs_buffer
        actions = self.replay_buffer.action_buffer
        dones = self.replay_buffer.done_buffer

        # Predict actions on the consequent state of each replay buffer example, apply noise to mitigate overfitting, and clip into expected range
        with torch.no_grad():
            projected_actions = self.actor.target_step(future_states)
            noise = torch.from_numpy(np.random.normal(0, self.action_noise, size=projected_actions.shape)).float()
            clipped_actions = torch.clamp(projected_actions + noise, -1, 1)
        
        #batched_state, batched_future_state, other_batches = self.batcher.mini_batch(initial_states, future_states, [rewards, actions, clipped_actions, dones])
        #batched_reward, batched_action, batched_clipped_action, batched_done = other_batches

        #Update the critic based on the replay butter example and processed projected actions on the consequent state
        self.critic.update(initial_states, actions, rewards, future_states,dones, clipped_actions)

        #When it's time to update the actor and critic targets, do so
        if self.number_of_updates % self.actor_update_delay == 0:
            #Predict actions for the antecedent state of each example
            current_actions = self.actor.batch_step(initial_states)
            #Calculate the loss (inverted reward for hill climbing)
            actor_loss = self.critic.get_actor_loss(initial_states, current_actions)
            #Update the actor
            self.actor.update(actor_loss)
            #Soft update the target networks
            self.critic.target_update()
        #Track the update state to enable delayed updates of the critic targets and the actor
        self.steps_since_update = 0
        self.number_of_updates += 1
