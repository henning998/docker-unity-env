from ensurepip import version
from os import times

import sys
import gym
import time
import numpy as np
import torch

import GA_evolver

# Create an instance from the class GA_evolver
from random import random

import mlagents
from mlagents_envs.environment import ActionTuple, BaseEnv
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


#from gym_unity.envs import UnityToGymWrapper
filename_env = "2DJumperBuildLinux/2DJumperBuildLinux"
population_size = 10
generations = 10
AI_instance = GA_evolver.GAevolver(population_size = population_size, generations=generations)

# Define model
input = 8       # Observation Space
hidden = [10]*5 # [5,5,5,5]
output = 3      # Action space

# Set evolution parameters
elitism = 2
top_best_actors_mutates = 5
random_mutation_percent = 5
amount_of_nonbest_actors_mutates = 0

AI_instance.set_model_parameters(input=input, hidden=hidden, output=output)
AI_instance.set_evolution_parameters(elitism=elitism, top_best_actors_mutates=top_best_actors_mutates, random_mutation_percent=random_mutation_percent, amount_of_nonbest_actors_mutates=amount_of_nonbest_actors_mutates)

AI_instance.generate_initial_jobs()

# Make Gym environment
#env = gym.make('CartPole-v1')
timesteps_per_episode = 100

channel = EngineConfigurationChannel()

env = UE(file_name=filename_env, seed=1, side_channels=[channel], no_graphics=True)
channel.set_configuration_parameters(time_scale = 100.0)

env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]

print("Number of observations : ", len(spec.observation_specs))

if spec.action_spec.is_continuous():
    print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
    print(f"There are {spec.action_spec.discrete_size} discrete actions")

#decision_steps, terminal_steps = env.get_steps(behavior_name)
#print(decision_steps.obs)

for generation in range(AI_instance.generations):
    gen_best_score = 0
    for actor in range(len(AI_instance.available_jobs)):
        env.reset()
        #decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1 # -1 indicates not yet tracking
        done = False # For the tracked_agent

        cum_reward = 0
        for timesteps in range(timesteps_per_episode):
            # Take a step
            env.step()
            
            # Move the simulation forward
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Track the first agent we see if not tracking 
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]
            if len(decision_steps) == 0:
                continue 

            # if tracked_agent in decision_steps: # The agent requested a decision
            #     cum_reward += decision_steps[tracked_agent].reward
            #     done = True
            #     break
            if tracked_agent in terminal_steps: # The agent terminated its episode
                goal_reward = terminal_steps[tracked_agent].reward
                if goal_reward > 0: # Goal Reached
                    cum_reward += goal_reward + (1 - (timesteps / timesteps_per_episode)) # Goal reward plus additional reward for how fast it reached the goal
                # Else no reward is given and cum reward stays at 0
                if goal_reward == 0 and timesteps < timesteps_per_episode-1:
                    cum_reward += -1
                #print("Reward = ", cum_reward)
                done = True
                break

            #print(decision_steps.obs[tracked_agent])
            
            action = AI_instance.available_jobs[actor][1](torch.from_numpy(decision_steps.obs[tracked_agent][0]))
            
            # Clamp the action
            action[action >= 1.0] = 1
            action[action <= -1.0] = 0

            # Set the actions
            action = action.cpu().detach().numpy()
            action = action.reshape(1,2)

            
            acton_tuple = ActionTuple()
            acton_tuple.add_continuous(action)
            #ac = spec.action_spec.random_action(len(decision_steps))
            env.set_action_for_agent(behavior_name, tracked_agent, acton_tuple)

        AI_instance.add_finished_job(actor, cum_reward)
        #print(cum_reward)
        if cum_reward > gen_best_score:
            gen_best_score = cum_reward
        #print("Reward for agent ", actor, ": ", cum_reward)
    AI_instance.on_generation()
    print("Generation ", generation, " Best Score = ", gen_best_score)

env.close()