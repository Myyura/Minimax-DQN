import os
import random
import numpy as np
import torch
import cvxopt

import gym
import matplotlib.pyplot as plt
import json

from agent import DQN_Agent
from train import train

def save_result(scores, filename: str, title: str):
    result_json = os.path.join('..', 'result', filename + '.json')
    with open(result_json, 'w') as f:
        json.dump(scores, f)

    fig, ax = plt.subplots(figsize=(16, 5))

    x = list(scores.keys())
    y = list(scores.values())
    y_mean = [np.average(y[i]) for i in range(len(y))]
    y_max = np.max(y, axis=1)
    y_min = np.min(y, axis=1)

    ax.plot(x, y_mean, 'o--', color='b', label = 'score')
    plt.fill_between(x, y_min, y_max, color='b', alpha=0.2)
    ax.legend(loc=2)
    ax.set_title(title)
    plt.savefig(os.path.join('..', 'result', filename + '.png'))

def train_cartpole(random_seed: int=42):
    '''Parameters'''
    max_step = 100000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128

    '''Standard'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)

    train_env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      device='cuda:0')
    
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = 'cuda_cartpole_' + 'standard_' + str(random_seed) + '_' + str(max_step)
    save_result(scores, result_filename, 'CartPole-v1, Standard-DQN')

    '''Minimax'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)

    train_env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      device='cuda:0')
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = 'cuda_cartpole_' + 'minimax_' + str(random_seed) + '_' + str(max_step)
    save_result(scores, result_filename, 'CartPole-v1, Minimax-DQN')

def train_lunarlander(random_seed: int=42):
    max_step = 700000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128

    '''Standard'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)
    
    train_env = gym.make('LunarLander-v2')
    eval_env = gym.make('LunarLander-v2')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = 'lunarlander_' + 'standard_' + str(random_seed) + '_' + str(max_step)
    save_result(scores, result_filename, 'LunarLander-v2, Standard-DQN')

    '''Minimax'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)

    train_env = gym.make('LunarLander-v2')
    eval_env = gym.make('LunarLander-v2')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = 'lunarlander_' + 'minimax_' + str(random_seed) + '_' + str(max_step)
    save_result(scores, result_filename, 'LunarLander-v2, Minimax-DQN')

def train_mountaincar(random_seed: int=42):
    max_step = 500000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128

    '''Standard'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)

    train_env = gym.make('MountainCar-v0')
    eval_env = gym.make('MountainCar-v0')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = 'tt_mountaincar_' + 'standard_' + str(random_seed) + '_' + str(max_step)
    save_result(scores, result_filename, 'MountainCar-v0, Standard-DQN')

    '''Minimax'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)

    train_env = gym.make('MountainCar-v0')
    eval_env = gym.make('MountainCar-v0')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = 'tt_mountaincar_' + 'minimax_' + str(random_seed) + '_' + str(max_step)
    save_result(scores, result_filename, 'MountainCar-v0, Minimax-DQN')

if __name__ == '__main__':
    train_cartpole(42)
    # train_lunarlander(42)
    # train_mountaincar(42)