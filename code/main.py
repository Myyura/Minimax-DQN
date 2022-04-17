import random
import numpy as np
import torch
import cvxopt

import gym
import matplotlib.pyplot as plt
import json

from agent import DQN_Agent
from train import train

def main(random_seed: int=42):
    # train_env = gym.make('MountainCar-v0')
    # eval_env = gym.make('MountainCar-v0')
    train_env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')
    train_env.reset(seed=random_seed)
    # eval_env.reset(seed=1234)
    train_env.action_space.seed(random_seed)
    # eval_env.action_space.seed(1234)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = DQN_Agent(state_dim, action_dim, 
                      batch_size=128, lr=1e-3, exploration_ratio=1.)

    # scores = train(
    #     agent, train_env, eval_env, 
    #     max_step=50000, warmup_step=256, train_step=1, eval_step=32,
    #     replay_buffer_size=5000, method='5-group-by-sampling', minimax_until=400.)
    scores = train(
        agent, train_env, eval_env, 
        max_step=50000, warmup_step=256, train_step=1, eval_step=32,
        replay_buffer_size=5000, method='5-group-by-sampling')

    # with open('../results/origin/cp_origin_64_ms50000_rbs5000_bs128_exp50_adam_seed42.json', 'w') as f:
    #     json.dump(scores, f)

    with open('./cp_minimax_0_evalfix_10000.json', 'w') as f:
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
    ax.set_title('limit: {}'.format(eval_env._max_episode_steps))
    # plt.savefig('../results/origin/cp_origin_64_ms50000_rbs5000_bs128_exp50_adam_seed42.png')
    plt.savefig('./cp_minimax_0_evalfix_10000.png')


if __name__ == '__main__':
    # set random seeds
    random_seed = 0

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True)
    cvxopt.setseed(random_seed)

    main(random_seed)