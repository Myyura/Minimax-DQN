import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from typing import Union
from cvxopt import matrix, solvers
from cvxopt.solvers import qp

from q_net import SimpleQNet
from utils import vector_to_grads, grads_to_vector

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        max_size: int) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dw = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, dw):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dw[self.ptr] = dw  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[index], 
            self.action[index], 
            self.reward[index], 
            self.next_state[index], 
            self.dw[index]
        )

    def ready_to_sample(self, warmup_size: int):
        return self.size > warmup_size

    def full(self):
        return self.size >= self.max_size

    def clear(self):
        self.ptr = 0
        self.size = 0


class GroupedReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        max_size_per_buffer: int,
        n: int) -> None:
        
        self.replay_buffers = [ReplayBuffer(state_dim, max_size_per_buffer) for _ in range(n)]
        self.n_buffer = n

        self.front = 0
        # self.back = -n

        self.size = 0

        self.max_size_per_buffer = max_size_per_buffer

    def add(self, state, action, reward, next_state, dw):
        if self.replay_buffers[self.front].full():
            self.front = (self.front + 1) % self.n_buffer
            if self.replay_buffers[self.front].full():
                self.replay_buffers[self.front].clear()
        
        self.replay_buffers[self.front].add(state, action, reward, next_state, dw)
        self.size = sum([buffer.size() for buffer in self.replay_buffers])

    @torch.no_grad()
    def sample(self, batch_size, warmup_size: Union[int, None]=None):
        if warmup_size is None:
            warmup_size = 2 * batch_size
        elif warmup_size < batch_size:
            raise ValueError('The argument "warmup_size" must be equal or greater than "batch_size"')

        states = []
        actions = []
        rewards = []
        next_states = []
        dws = []
        for replay_buffer in self.replay_buffers:
            if replay_buffer.ready_to_sample(warmup_size):
                s, a, r, s_prime, dw = replay_buffer.sample(self.batch_size)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(s_prime)
                dws.append(dw)
        
        return (states, actions, rewards, next_states, dws, len(states))

    def ready_to_sample(self, warmup_size: int):
        return self.size > warmup_size


class DQN_Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float=0.99,
        lr: float=5e-4,
        batch_size: int=128,
        exploration_ratio: float=0.5,
        eval_step: int=32,
        device: str='cpu') -> None:

        self.q_net = SimpleQNet(state_dim, action_dim).to(device)
        # The only trick we use is target network
        self.target_net = SimpleQNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.exploration_ratio = exploration_ratio
        self.action_dim = action_dim
        self.device = device
        self.training_step = 0
        self.eval_step = eval_step

    @torch.no_grad()
    def select_action(self, state, deterministic):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.device)
        if deterministic:
            a = self.q_net(state).argmax().item()
        else:
            if np.random.rand() < self.exploration_ratio:
                a = np.random.randint(0, self.action_dim)
            else:
                a = self.q_net(state).argmax().item()
        return a

    def exploration_decay(self, min_ratio: float=0.05, decay_ratio: float=0.99):
        if self.exploration_ratio <= min_ratio:
            self.exploration_ratio = min_ratio
        else:
            self.exploration_ratio *= decay_ratio

    def _dqn_loss(self, s, a, r, s_prime, dw):
        '''Compute the target Q value'''
        with torch.no_grad():
            max_q_prime = self.target_net(s_prime).max(1)[0].unsqueeze(1)
            '''Avoid impacts caused by reaching max episode steps'''
            target_Q = r + (1 - dw) * self.gamma * max_q_prime # dw: die and win
            # target_Q = r + self.gamma * max_q_prime

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)
        return q_loss
    
    def _train_standard(self, s, a, r, s_prime, dw):
        '''The standard Deep Q-Learning'''
        # update the target network every fixed steps
        if self.training_step % self.eval_step == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.q_net.state_dict())

        q_loss = self._dqn_loss(s, a, r, s_prime, dw)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        self.training_step += 1
    
    def _transition_to_device(self, s, a, r, s_prime, dw):
        return (
            torch.tensor(s, dtype=torch.float).to(self.device),
            torch.tensor(a, dtype=torch.int64).to(self.device),
            torch.tensor(r, dtype=torch.float).to(self.device),
            torch.tensor(s_prime, dtype=torch.float).to(self.device),
            torch.tensor(dw, dtype=torch.int64).to(self.device)
        )

    def train_standard(self, replay_buffer: ReplayBuffer):
        s, a, r, s_prime, dw = \
            self._transition_to_device(*replay_buffer.sample(self.batch_size))
        self._train_standard(s, a, r, s_prime, dw)

    def train_minimax_group_by_step(self, replay_buffer: GroupedReplayBuffer):
        # update the target network every fixed steps
        if self.training_step % self.eval_step == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.q_net.state_dict())

        states, actions, rewards, next_states, dws, n = replay_buffer.sample(self.batch_size)
        if n < 1:
            print('Wait for enough replays')
            return
        elif n == 1:
            # warmup, using standard trainning method
            s, a, r, s_prime, dw = \
                self._transition_to_device(states[0], actions[0], rewards[0], next_states[0], dws[0])
            self._train_standard(s, a, r, s_prime, dw)
        else:
            # train with minimax deep Q-learning
            grouped_grads = []
            grouped_loss = []
            for i in range(n):
                s, a, r, s_prime, dw = \
                    self._transition_to_device(states[i], actions[i], rewards[i], next_states[i], dws[i])
                '''Compute the target Q value'''
                q_loss = self._dqn_loss(s, a, r, s_prime, dw)
                self.optimizer.zero_grad()
                q_loss.backward()
                grads = grads_to_vector(self.q_net.parameters())

                grouped_grads.append(grads.detach().numpy())
                grouped_loss.append(q_loss.detach().numpy())

            grads = grad_by_minimax(
                        torch.tensor(grouped_grads, dtype=torch.float), 
                        torch.tensor(grouped_loss, dtype=torch.float)
                    )
            vector_to_grads(
                torch.tensor(grads, dtype=torch.float).to(self.device), 
                self.q_net.parameters()
            )

            self.optimizer.step()
            self.training_step += 1

    def train_minimax_group_by_sampling(self, replay_buffer: ReplayBuffer, n: int):
        # update the target network every fixed steps
        if self.training_step % self.eval_step == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.q_net.state_dict())

        grouped_grads = []
        grouped_loss = []
        for i in range(n):
            s, a, r, s_prime, dw = \
                self._transition_to_device(*replay_buffer.sample(self.batch_size))
            q_loss = self._dqn_loss(s, a, r, s_prime, dw)
            self.optimizer.zero_grad()
            q_loss.backward()
            grads = grads_to_vector(self.q_net.parameters())

            grouped_grads.append(grads.cpu().detach().numpy())
            grouped_loss.append(q_loss.cpu().detach().numpy())
        
        grouped_loss = np.array(grouped_loss)
        grads = grad_by_minimax(
                    torch.tensor(grouped_grads, dtype=torch.float), 
                    torch.tensor(grouped_loss, dtype=torch.float)
                )
        vector_to_grads(
            torch.tensor(grads, dtype=torch.float).to(self.device), 
            self.q_net.parameters()
        )

        self.optimizer.step()
        self.training_step += 1

    def train(self, replay_buffer: Union[GroupedReplayBuffer, ReplayBuffer], method: str='standard'):
        if method == 'standard':
            self.train_standard(replay_buffer)
        elif 'group-by-step' in method:
            self.train_minimax_group_by_step(replay_buffer)
        else:
            pass

    def save(self, model_path: str):
        torch.save(self.q_net.state_dict(), model_path)

    def load(self, model_path: str):
        self.q_net.load_state_dict(torch.load(model_path))


def grad_by_minimax(grouped_grads: torch.Tensor, grouped_loss: torch.Tensor):
    r""" Calculate grads by minimax algorithm

    Args: 
        grouped_grads (Tensor[n, m]): grads vector of n groups
        grouped_loss (Tensor[n, 1]): loss vector of n groups

    Returns:

    """

    n, m = grouped_grads.shape

    r"""
    minimze (1/2)xPx + qx          |         min (1/2)xDDx - fx
    subject to Gx <= h             |         subject to \sum_i^n x_i = 1
               Ax = b              |                    x_i >= 0

    D: grouped_grads, tensor of shape (n, m)
    """

    # D and h
    D = matrix(grouped_grads.numpy().astype(np.float64))
    f = matrix(grouped_loss.numpy().reshape(-1).astype(np.float64))

    P = D * D.T
    q = -f

    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(np.ones([1, n]))
    b = matrix(np.ones([1]))

    solvers.options['show_progress'] = False
    res = qp(P, q, G, h, A, b)
    d = np.array(D).T.dot(np.array(res['x']))[:, 0]

    return d