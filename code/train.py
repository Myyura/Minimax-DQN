import numpy as np
import gym

from collections import deque
from typing import Union
from agent import DQN_Agent, GroupedReplayBuffer, ReplayBuffer

def evaluate(env: gym.Env, agent: DQN_Agent, turns: int=100):
    scores = []
    for i in range(turns):
        score = 0
        # eval_env.reset(seed=1234)
        s, done = env.reset(seed=10000+i), False
        env.action_space.seed(10000+i)
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
    last3_score = deque([0, 0, 0], maxlen=3)
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
            # change the reward function of MountainCar-v0
            if train_env.spec.id == 'MountainCar-v0':
                r = 20 if s_prime[0] >= 0.5 else abs(s_prime[0] - s[0])

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
                    if minimax_until is not None \
                        and last3_score is not None \
                        and not use_standard \
                        and np.average(last3_score) >= minimax_until:
                        use_standard = True

                    if use_standard:
                        agent.train_standard(replay_buffer)
                    else:
                        agent.train_minimax_group_by_sampling(replay_buffer, n_sample)

                # exploration decay
                agent.exploration_decay()

            '''Evaluate'''
            if total_step % eval_step == 0:
                scores[total_step] = evaluate(eval_env, agent)
                if last3_score is not None:
                    last3_score.append(np.average(scores[total_step]))
        total_episode += 1

        '''Log'''
        print('Steps: {}, Episode: {}, Score: {}, Exploration Ratio: {}'.format(total_step, total_episode, ep_r, agent.exploration_ratio))

    return scores