import os
import random
from unittest import result
import numpy as np
import torch
import cvxopt

import gym
import torch.nn as nn

from typing import Optional, Union
from ddpg_agent import DDPG_Agent
from replay_buffer import ReplayBuffer, ProportionalPrioritizedReplayBuffer
from utils import GaussianExplorationNoise, OrnsteinUhlenbeckActionNoise, save_result

def evaluate(env: gym.Env, agent: DDPG_Agent, turns: int=50):
    # eval mode
    agent.eval_mode()

    scores = []
    for i in range(turns):
        score = 0
        s, done = env.reset(seed=10000+i), False
        while not done:
            a = agent.select_action(s, exploration_noise=None)
            # print(a)
            s_prime, r, done, info = env.step(a)
            # print(s_prime, r)
            score += r
            s = s_prime
        
        scores.append(score)

    # back to training mode
    agent.train_mode()
    return scores

def train_ddpg(
    agent: DDPG_Agent, 
    train_env: gym.Env,
    eval_env: gym.Env,
    max_step: int, 
    warmup_step: int,
    train_step: int,
    eval_step: int,
    replay_buffer_size: int,
    method: str='standard',
    exploration: str='normal'):
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    if exploration == 'normal':
        exploration_noise = GaussianExplorationNoise()
    elif exploration == 'ou':
        exploration_noise = OrnsteinUhlenbeckActionNoise()

    replay_buffers = []
    if method == 'standard':
        replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=replay_buffer_size)
        replay_buffers.append(replay_buffer)
    elif 'group-by-sampling' in method:
        n_sample = int(method.split('-')[0])
        replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=replay_buffer_size)
        replay_buffers.append(replay_buffer)

    scores = {}
    total_step = 0
    total_episode = 0
    max_score = -999999
    q_loss, policy_loss = -999999, -999999
    while total_step < max_step:
        s, done, ep_r, ep_step = train_env.reset(), False, 0, 0

        while not done:
            ep_step += 1
            total_step += 1

            if total_step > warmup_step:
                a = agent.select_action(s, exploration_noise=exploration_noise)
            else:
                a = train_env.action_space.sample()
                # print('sample action: {}'.format(a))

            s_prime, r, done, info = train_env.step(a)
            # print(s_prime, r)

            '''Avoid impacts caused by reaching max episode steps'''
            dw = 1 if done else 0

            for replay_buffer in replay_buffers:
                replay_buffer.add(s, a, r, s_prime, dw)
            
            s = s_prime
            ep_r += r

            '''Train'''
            if total_step > warmup_step and total_step % train_step == 0:
                if method == 'standard':
                    q_loss, policy_loss = agent.train_standard(replay_buffers[0])
                elif 'group-by-sampling' in method:
                    q_loss, policy_loss = agent.train_minimax_group_by_sampling(replay_buffers[0], n_sample)

            '''Evaluate'''
            if total_step % eval_step == 0:
                scores[total_step] = evaluate(eval_env, agent, turns=10)
                # save the best model
                ave_score = np.average(scores[total_step])
                if ave_score > max_score:
                    model_path = os.path.join('..', 'model', str(train_env.spec.id) + '_' + method + '_best.pth')
                    # agent.save(model_path)
                    max_score = ave_score
                print('Steps: {}, Episode: {}, Eval Score: {}, Max Score: {}, Q Loss: {}, Policy Loss: {}'.format(total_step, total_episode, ave_score, max_score, q_loss, policy_loss))
                

        total_episode += 1

        '''Log'''
        # print('Steps: {}, Episode: {}, Score: {}, Exploration Ratio: {}'.format(total_step, total_episode, ep_r, agent.exploration_ratio))

    return scores

def train(
    env: str='Pendulum-v1', 
    method: str='standard',
    exploration:str='normal',
    lr_policy: float=5e-4,
    lr_q: float=1e-3,
    train_step: int=4,
    eval_step: int=32,
    syn_step: int=4,
    warmup_step: int=1000,
    batchsize: int=128,
    max_step: int=400000, 
    replay_buffer_size: int=20000, 
    actor: Union[str, nn.Module]='simple_mlp_actor', 
    critic: Union[str, nn.Module]='simple_mlp_critic', 
    result_filename: Optional[str]=None,
    random_seed: int=42,
    device='cpu'):

    print('Env: {} \nLR (Policy): {}, LR (Q): {}, BatchSize: {}, MaxStep: {}, BufferSize: {}, RandomSeed: {}, Method: {}, Exploration: {}'.format(env, 
        lr_policy, lr_q, batchsize, max_step, replay_buffer_size, random_seed, method, exploration))
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cvxopt.setseed(random_seed)
    train_env = gym.make(env)
    eval_env = gym.make(env)
    train_env.reset(seed=random_seed)
    train_env.action_space.seed(random_seed)
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    action_low = train_env.action_space.low[0]
    action_high = train_env.action_space.high[0]

    agent = DDPG_Agent(
        actor=actor,
        critic=critic,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        gamma=0.99,
        polyak=0.995,
        lr_actor=lr_policy,
        lr_critic=lr_q,
        batch_size=batchsize,
        syn_step=syn_step,
        device=device
    )

    scores = train_ddpg(
        agent=agent, 
        train_env=train_env, 
        eval_env=eval_env, 
        max_step=max_step,
        warmup_step=warmup_step,
        train_step=train_step,
        eval_step=eval_step,
        replay_buffer_size=replay_buffer_size,
        method=method)
        
    if result_filename is None:
        result_filename = env + '_' + method + '_' + exploration + '_' + str(batchsize) + '_' + str(max_step) + '_' + str(replay_buffer_size) + str(random_seed)
    save_result(scores, result_filename, env + ', ' + method)


if __name__ == '__main__':
    # train(method='5-group-by-sampling')
    # train(env='MountainCarContinuous-v0', exploration='ou')
    train(env='MountainCarContinuous-v0', exploration='ou', method='5-group-by-sampling')