import numpy as np
import torch
import os
import functools
import time

from tqdm import trange

from live import live
from environment import MountainCarEnv
from agents import DQNAgent
from agents import EnsembleDQNAgent
from agents import BootDQNAgent
from agents import mountaincar_reward_function
from feature import MountainCarIdentityFeature

def DQN(config):
    reward_path = './'+ str(config.mode) +'/results/'+ time.strftime("%Y%m%d-%H%M%S") +'/'
    agent_path = './'+ str(config.mode) +'/agents/' + time.strftime("%Y%m%d-%H%M%S") +'/'
    if not os.path.exists(reward_path):
        os.makedirs(reward_path)
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)
    env = MountainCarEnv()
    number_seeds = config.number_seeds
    for seed in trange(number_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = DQNAgent(
            action_set=[0, 1, 2],
            reward_function=functools.partial(mountaincar_reward_function, reward_type='sparse'),
            feature_extractor=MountainCarIdentityFeature(),
            hidden_dims=[[50, 50],[10,10,5],[50,10]][config.hidden_idx],
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            num_batches=config.num_batches,
            starts_learning=config.starts_learning,
            final_epsilon=config.final_epsilon,
            discount=config.discount,
            target_freq=config.target_freq)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            num_episodes=config.num_episodes,
            max_timesteps=config.max_timesteps,
            verbose = config.verbose,
            print_every=config.print_every)

        file_name = '|'.join(['dqn', str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))
        env.close()

def EnsembleDQN(config):
    reward_path = './'+ str(config.mode) +'/results/'+ time.strftime("%Y%m%d-%H%M%S") +'/'
    agent_path = './'+ str(config.mode) +'/agents/' + time.strftime("%Y%m%d-%H%M%S") +'/'
    if not os.path.exists(reward_path):
        os.makedirs(reward_path)
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)
    env = MountainCarEnv()
    number_seeds = config.number_seeds
    for seed in trange(number_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = EnsembleDQNAgent(
            action_set=[0, 1, 2],
            reward_function=functools.partial(mountaincar_reward_function, reward_type='sparse'),
            feature_extractor=MountainCarIdentityFeature(),
            hidden_dims=[[50, 50],[10,10,5],[50,10]][config.hidden_idx],
            num_ensemble=config.num_ensemble,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            num_batches=config.num_batches,
            starts_learning=config.starts_learning,
            final_epsilon=config.final_epsilon,
            discount=config.discount,
            target_freq=config.target_freq)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            num_episodes=config.num_episodes,
            max_timesteps=config.max_timesteps,
            verbose = config.verbose,
            print_every=config.print_every)

        file_name = '|'.join(['dqn', str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))
        env.close()

def BootDQN(config):
    reward_path = './'+ str(config.mode) +'/results/'+ time.strftime("%Y%m%d-%H%M%S") +'/'
    agent_path = './'+ str(config.mode) +'/agents/' + time.strftime("%Y%m%d-%H%M%S") +'/'
    if not os.path.exists(reward_path):
        os.makedirs(reward_path)
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)
    env = MountainCarEnv()
    number_seeds = config.number_seeds
    for seed in trange(number_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = BootDQNAgent(
            action_set=[0, 1, 2],
            reward_function=functools.partial(mountaincar_reward_function, reward_type='sparse'),
            feature_extractor=MountainCarIdentityFeature(),
            hidden_dims=[[50, 50],[10,10,5],[50,10]][config.hidden_idx],
            num_ensemble=config.num_ensemble,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            num_batches=config.num_batches,
            starts_learning=config.starts_learning,
            final_epsilon=config.final_epsilon,
            discount=config.discount,
            target_freq=config.target_freq)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            num_episodes=config.num_episodes,
            max_timesteps=config.max_timesteps,
            verbose = config.verbose,
            print_every=config.print_every)

        file_name = '|'.join(['dqn', str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))
        env.close()

if __name__ == '__main__':

    reward_path = './results/'
    agent_path = './agents/'

    # env = CartpoleEnv(swing_up=True)
    env = MountainCarEnv()


    # train dqn agents
    number_seeds = 1
    for seed in trange(number_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = EnsembleDQNAgent(
            action_set=[0, 1, 2],
            # reward_function=functools.partial(cartpole_reward_function, reward_type='sparse'),
            reward_function=functools.partial(mountaincar_reward_function, reward_type='sparse'),
            feature_extractor=MountainCarIdentityFeature(),
            hidden_dims=[50, 50],
            num_ensemble=10,
            learning_rate=5e-4,
            buffer_size=50000,
            batch_size=64,
            num_batches=100,
            starts_learning=5000,
            final_epsilon=0.02,
            discount=0.99,
            target_freq=10,
            verbose=False,
            print_every=10)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            num_episodes=1000,
            max_timesteps=500,
            verbose=True,
            print_every=50)

        file_name = '|'.join(['dqn', str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))
        env.close()
