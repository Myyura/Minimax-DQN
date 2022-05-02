import os
import numpy as np
import gym

from typing import Union
from agent import DQN_Agent, GroupedReplayBuffer, ReplayBuffer

def evaluate(env: gym.Env, agent: DQN_Agent, turns: int=100):
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
    method: str='standard',
    minimax_until: Union[float, None] = None):
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    if method == 'standard':
        replay_buffer = ReplayBuffer(state_dim=state_dim, max_size=replay_buffer_size)
    elif 'group-by-step' in method:
        n_buffer = int(method.split('-')[0])
        replay_buffer = GroupedReplayBuffer(
                            state_dim=state_dim, 
                            max_size_per_buffer=replay_buffer_size//n_buffer, 
                            n=n_buffer)
    elif 'group-by-sampling' in method:
        n_sample = int(method.split('-')[0])
        replay_buffer = ReplayBuffer(state_dim=state_dim, max_size=replay_buffer_size)

        use_standard = False

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

            replay_buffer.add(s, a, r, s_prime, dw)
            
            s = s_prime
            ep_r += r

            '''Train'''
            if total_step > warmup_step and total_step % train_step == 0:
                if method == 'standard':
                    agent.train_standard(replay_buffer)
                elif 'group-by-step' in method:
                    pass
                elif 'group-by-sampling' in method:
                    if use_standard:
                        agent.train_standard(replay_buffer)
                    else:
                        agent.train_minimax_group_by_sampling(replay_buffer, n_sample)

                # exploration decay
                agent.exploration_decay()

            '''Evaluate'''
            if total_step % eval_step == 0:
                scores[total_step] = evaluate(eval_env, agent)
                # save the best model
                ave_score = np.average(scores[total_step])
                if ave_score > max_score:
                    model_path = os.path.join('..', 'model', str(train_env.spec.id) + '_' + method + '_best.pth')
                    agent.save(model_path)
                    max_score = ave_score

        total_episode += 1

        '''Log'''
        print('Steps: {}, Episode: {}, Score: {}, Exploration Ratio: {}'.format(total_step, total_episode, ep_r, agent.exploration_ratio))

    return scores