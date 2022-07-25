import os
import random
import numpy as np
import torch
import cvxopt
import gym

from dqn_agent import DQN_Agent
from ddpg_agent import DDPG_Agent
from utils import save_result
from replay_buffer import ReplayBuffer, ProportionalPrioritizedReplayBuffer
from model import QNet128, QNet64

def evaluate(env: gym.Env, agent: DQN_Agent, turns: int=50):
    # eval mode
    agent.eval_mode()

    scores = []
    for i in range(turns):
        score = 0
        s, done = env.reset(seed=10000+i), False
        while not done:
            a = agent.select_action(s, deterministic=True)
            s_prime, r, done, info = env.step(a)

            score += r
            s = s_prime
        
        scores.append(score)

    # back to training mode
    agent.train_mode()
    return scores


def train(
    agent: DQN_Agent, 
    train_env: gym.Env,
    eval_env: gym.Env,
    max_step: int, 
    warmup_step: int,
    train_step: int,
    eval_step: int,
    replay_buffer_size: int,
    method: str='standard'):
    state_dim = train_env.observation_space.shape[0]
    # action_dim = train_env.action_space.n

    replay_buffers = []
    if method == 'standard':
        replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=1, max_size=replay_buffer_size)
        replay_buffers.append(replay_buffer)
    elif 'group-by-sampling' in method:
        n_sample = int(method.split('-')[0])
        replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=1, max_size=replay_buffer_size)
        replay_buffers.append(replay_buffer)
    elif 'proportional-per' in method:
        replay_buffer = ProportionalPrioritizedReplayBuffer(
            state_dim=state_dim, max_size=replay_buffer_size, alpha=0.6)
        beta_proportional = 0.4 # parameter setting following the original paper of PER
        replay_buffers.append(replay_buffer)
    elif 'rank-based-per' in method:
        pass
    elif 'group-by-per' in method:
        pass

    scores = {}
    total_step = 0
    total_episode = 0
    max_score = -999999
    while total_step < max_step:
        s, done, ep_r, ep_step = train_env.reset(), False, 0, 0

        while not done:
            ep_step += 1
            total_step += 1

            if total_step > warmup_step:
                a = agent.select_action(s, deterministic=False)
            else:
                a = train_env.action_space.sample()

            s_prime, r, done, info = train_env.step(a)

            '''Avoid impacts caused by reaching max episode steps'''
            dw = 1 if done else 0

            for replay_buffer in replay_buffers:
                replay_buffer.add(s, a, r, s_prime, dw)
            
            s = s_prime
            ep_r += r

            '''Train'''
            if total_step > warmup_step and total_step % train_step == 0:
                if method == 'standard':
                    agent.train_standard(replay_buffers[0])
                elif 'group-by-sampling' in method:
                    agent.train_minimax_group_by_sampling(replay_buffers[0], n_sample)
                elif 'proportional-per' in method:
                    agent.train_with_proportional_per(replay_buffers[0], beta_proportional)

                # exploration decay
                agent.exploration_decay()

            '''Evaluate'''
            if total_step % eval_step == 0:
                scores[total_step] = evaluate(eval_env, agent, turns=50)
                # save the best model
                ave_score = np.average(scores[total_step])
                if ave_score > max_score:
                    model_path = os.path.join('..', 'model', str(train_env.spec.id) + '_' + method + '_best.pth')
                    agent.save(model_path)
                    max_score = ave_score
                print('Steps: {}, Episode: {}, Eval Score: {}, Max Score: {}, Exploration Ratio: {}'.format(total_step, total_episode, ave_score, max_score, agent.exploration_ratio))
                

        total_episode += 1

        '''Log'''
        # print('Steps: {}, Episode: {}, Score: {}, Exploration Ratio: {}'.format(total_step, total_episode, ep_r, agent.exploration_ratio))

    return scores

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
    max_step = 200000
    replay_buffer_size = max_step // 20
    lr = 5e-4
    batch_size = 128
    qnet = 'qnet128'
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
    agent = DQN_Agent(state_dim, action_dim, q_net=qnet,
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    
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
    agent = DQN_Agent(state_dim, action_dim, q_net=qnet,
                      batch_size=batch_size, lr=lr, exploration_ratio=1.)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = str(qnet) + 'ttt_correct_cartpole_minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'CartPole-v1, Minimax-DQN')

def train_lunarlander_standard(random_seed: int=42):
    max_step = 1000000
    replay_buffer_size = max_step // 20
    lr = 5e-4
    batch_size = 128
    qnet = 'qnet128'
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
                      q_net=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='standard')

    result_filename = '1_correct_lunarlander50_standard_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'LunarLander-v2, Standard-DQN')

def train_lunarlander_minimax(random_seed: int=42):
    max_step = 1000000
    replay_buffer_size = max_step // 20
    lr = 5e-4
    batch_size = 128
    qnet = 'qnet128'
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
                      q_net=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = '1_correct_lunarlander50_minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'LunarLander-v2, Minimax-DQN')

def train_mountaincar(random_seed: int=42):
    max_step = 1000000
    replay_buffer_size = max_step // 20
    lr = 5e-4
    batch_size = 128
    qnet = 'qnet64'
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
                      q_net=qnet)
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
                      q_net=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = '2_correct_mountaincar_' + 'minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    print(result_filename)
    save_result(scores, result_filename, 'MountainCar-v0, Minimax-DQN')

def train_acrobot(random_seed: int=42):
    max_step = 60000
    replay_buffer_size = max_step // 20
    lr = 5e-4
    batch_size = 128
    qnet = 'qnet64'
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
                      q_net=qnet)
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
                      q_net=qnet)
    scores = train(
        agent, train_env, eval_env,
        max_step=max_step, warmup_step=batch_size*2, train_step=4, eval_step=32,
            replay_buffer_size=replay_buffer_size, method='5-group-by-sampling')

    result_filename = str(qnet) + '_correct_acrobot_minimax_' + str(random_seed) + '_' + str(max_step) + '_' + str(replay_buffer_size)
    save_result(scores, result_filename, 'Acrobot-v1, Minimax-DQN')

if __name__ == '__main__':
    # train_cartpole_proportional_per(42)
    # train_cartpole(42)
    # train_lunarlander_standard(42)
    # train_lunarlander_minimax(42)
    # train_mountaincar(42)
    train_acrobot(42)