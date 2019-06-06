import numpy as np
import torch
import os
import functools
import gym


from agents import DQNAgent
from rlsvi import RLSVIIncrementalTDAgent
from agents import mountaincar_reward_function
from feature import MountainCarIdentityFeature

np.random.seed(338)
torch.manual_seed(338)

def log(logfile, iteration, rewards):
    """Function that logs the reward statistics obtained by the agent.

    Args:
        logfile: File to log reward statistics.
        iteration: The current iteration.
        rewards: Array of rewards obtained in the current iteration.
    """
    log_string = '{} {} {} {}'.format(
        iteration, np.min(rewards), np.mean(rewards), np.max(rewards))
    print(log_string)

    with open(logfile, 'a') as f:
        f.write(log_string + '\n')

def DQN(config):
    log_path = './'+ str(config.mode) +'/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    env = gym.make('MountainCar-v0')
    env_eval = gym.make('MountainCar-v0')

    agent = DQNAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(mountaincar_reward_function),
        feature_extractor=MountainCarIdentityFeature(),
        hidden_dims= [64],
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        num_batches=config.num_batches,
        starts_learning=config.starts_learning,
        final_epsilon=config.final_epsilon,
        discount=config.discount,
        target_freq=config.target_freq)

    reward_data = []
    for episode in range(config.num_episodes):
        for train_it in range(config.train_iterations):
            agent.reset_cumulative_reward()
            observation_history = [(env.reset(), False)]
            action_history = []
        
            done = False
            while not done:
                action = agent.act(observation_history, action_history)
                observation, _, done,_ = env.step(action)
                action_history.append(action)
                observation_history.append((observation, done))
                done = done
            agent.update_buffer(observation_history, action_history)
            agent.learn_from_buffer()

        eval_rewards = []
        for eval_it in range(config.eval_iterations):
            agent.test_mode = True
            agent.reset_cumulative_reward()
            observation_history = [(env_eval.reset(), False)]
            action_history = []
        
            done = False
            while not done:
                action = agent.act(observation_history, action_history)
                observation, _, done,_ = env_eval.step(action)
                action_history.append(action)
                observation_history.append((observation, done))
                done = done
            agent.update_buffer(observation_history, action_history)
            eval_rewards.append(agent.cummulative_reward)
            agent.test_mode = False

        if episode % config.print_every == 0:
            log(log_path + 'log.txt', episode, eval_rewards)
            agent.save(path= log_path +'agent.pt')
            
        reward_data.append(eval_rewards)
    reward_data = np.asarray(reward_data)
    np.save(log_path + 'rewards', reward_data)


def RLSVI(config):
    log_path = './'+ str(config.mode) + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    env = gym.make('MountainCar-v0')
    env_eval = gym.make('MountainCar-v0')

    agent = RLSVIIncrementalTDAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(mountaincar_reward_function),
        feature_extractor=MountainCarIdentityFeature(),
        hidden_dims=[64],
        prior_variance = config.prior_variance,
        noise_variance = config.noise_variance,
        prior_network = True,
        num_ensemble=config.num_ensemble,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        num_batches=config.num_batches,
        starts_learning=config.starts_learning,
        discount=config.discount,
        target_freq=config.target_freq)

    reward_data = []
    for episode in range(config.num_episodes):
        for train_it in range(config.train_iterations):
            agent.reset_cumulative_reward()
            observation_history = [(env.reset(), False)]
            action_history = []

            done = False
            while not done:
                action = agent.act(observation_history, action_history)
                observation, _, done,_ = env.step(action)
                action_history.append(action)
                observation_history.append((observation, done))
                done = done
            agent.update_buffer(observation_history, action_history)
            agent.learn_from_buffer()

        eval_rewards = []
        for eval_it in range(config.eval_iterations):
            agent.test_mode = True
            agent.reset_cumulative_reward()
            observation_history = [(env_eval.reset(), False)]
            action_history = []

            done = False
            while not done:
                action = agent.act(observation_history, action_history)
                observation, _, done,_ = env_eval.step(action)
                action_history.append(action)
                observation_history.append((observation, done))
                done = done
            agent.update_buffer(observation_history, action_history)
            eval_rewards.append(agent.cummulative_reward)
            agent.test_mode = False

        if episode % config.print_every == 0:
            log(log_path + 'log.txt', episode, eval_rewards)
            agent.save(path= log_path +'agent.pt')

        reward_data.append(eval_rewards)
    reward_data = np.asarray(reward_data)
    np.save(log_path + 'rewards', reward_data)

