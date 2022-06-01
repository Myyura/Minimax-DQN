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

def train_cartpole_proportional_per(random_seed: int=42):
    '''Parameters'''
    max_step = 100000
    replay_buffer_size = 2**13
    lr = 5e-4
    batch_size = 128

    '''Proportional-PER'''
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
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='proportional-per')

    result_filename = 'correct_cartpole_' + 'proportional_per_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'CartPole-v1, Proportional-PER-DQN')

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
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = 'correct_cartpole_' + 'standard_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
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
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = 'correct_cartpole_' + 'minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'CartPole-v1, Minimax-DQN')
    
    
def train_cartpole(random_seed: int=42):
    '''Parameters'''
    max_step = 100000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128
    qnet = 1
    print(max_step, replay_buffer_size, lr, batch_size, qnet, 'CartPole-v1', random_seed)

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
                      qnet=qnet)
    
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = str(qnet) + 'ttt_correct_cartpole_standard_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
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
                      qnet=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = str(qnet) + 'ttt_correct_cartpole_minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'CartPole-v1, Minimax-DQN')

def train_lunarlander_standard(random_seed: int=42):
    max_step = 500000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128
    qnet = 1
    print(max_step, replay_buffer_size, lr, batch_size, qnet, 'LunarLander-v2', random_seed)

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
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      qnet=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = '1_correct_lunarlander50_standard_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'LunarLander-v2, Standard-DQN')

def train_lunarlander_minimax(random_seed: int=42):
    max_step = 500000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128
    qnet = 1
    print(max_step, replay_buffer_size, lr, batch_size, qnet, 'LunarLander-v2', random_seed)

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
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      qnet=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = '1_correct_lunarlander50_minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'LunarLander-v2, Minimax-DQN')

def train_mountaincar(random_seed: int=42):
    max_step = 500000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128
    qnet = 2
    print(max_step, replay_buffer_size, lr, batch_size, qnet, 'Mountaincar-v2', random_seed)

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
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      qnet=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = '2_correct_mountaincar_' + 'standard_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    print(result_filename)
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
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      qnet=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = '2_correct_mountaincar_' + 'minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    print(result_filename)
    save_result(scores, result_filename, 'MountainCar-v0, Minimax-DQN')

def train_acrobot(random_seed: int=42):
    max_step = 20000
    replay_buffer_size = max_step // 10
    lr = 5e-4
    batch_size = 128
    qnet = 2
    print(max_step, replay_buffer_size, lr, batch_size, qnet, 'Acrobot-v2', random_seed)
    
    '''Standard'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)

    train_env = gym.make('Acrobot-v1')
    eval_env = gym.make('Acrobot-v1')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      qnet=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = str(qnet) + '_correct_acrobot_standard_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'Acrobot-v1, Standard-DQN')

    '''Minimax'''
    # reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)

    train_env = gym.make('Acrobot-v1')
    eval_env = gym.make('Acrobot-v1')
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=batch_size, lr=lr, exploration_ratio=1.,
                      qnet=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = str(qnet) + '_correct_acrobot_minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'Acrobot-v1, Minimax-DQN')

if __name__ == '__main__':
    # train_cartpole_proportional_per(42)
    train_cartpole(42)
    # train_lunarlander_standard(42)
    # train_lunarlander_minimax(42)
    # train_mountaincar(42)
    # train_acrobot(42)