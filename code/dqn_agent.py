import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from model import QNet64, QNet128
from replay_buffer import ReplayBuffer, ProportionalPrioritizedReplayBuffer
from utils import vector_to_grads, grads_to_vector, grad_by_minimax


class DQN_Agent:
    '''Standard Double DQN Agent'''
    def __init__(
        self,
        state_dim: Optional[int]=None,
        action_dim: Optional[int]=None,
        q_net: Union[str, nn.Module]='qnet64',
        target_net: Optional[nn.Module]=None,
        gamma: float=0.99,
        lr: float=5e-4,
        batch_size: int=128,
        exploration_ratio: float=0.5,
        syn_step: int=32,
        device: str='cpu') -> None:

        if isinstance(q_net, str):
            if q_net == 'qnet64':
                self.q_net = QNet64(state_dim, action_dim).to(device)
                self.target_net = QNet64(state_dim, action_dim).to(device)
            elif q_net == 'qnet128':
                self.q_net = QNet128(state_dim, action_dim).to(device)
                self.target_net = QNet128(state_dim, action_dim).to(device)
        elif isinstance(q_net, nn.Module):
            self.q_net = q_net.to(device)
            self.target_net = target_net.to(device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.exploration_ratio = exploration_ratio
        self.action_dim = action_dim
        self.device = device
        self.training_step = 0
        self.syn_step = syn_step

    def train_mode(self):
        self.q_net.train()

    def eval_mode(self):
        self.q_net.eval()

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

    def _update_target_network(self):
        # update the target network every fixed steps
        if self.training_step % self.syn_step == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _dqn_loss(self, s, a, r, s_prime, dw, weight=None):
        # Compute the target Q value
        with torch.no_grad():
            next_a = self.q_net(s_prime).argmax(1)
            max_q_prime = self.target_net(s_prime).gather(1, next_a.unsqueeze(-1))
            # max_q_prime = self.target_net(s_prime).max(1)[0].unsqueeze(1)
            '''Avoid impacts caused by reaching max episode steps'''
            target_q = r + (1 - dw) * self.gamma * max_q_prime # dw: die and win
            # target_q = r + self.gamma * max_q_prime

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        td_error = target_q - current_q_a
        if weight is None:
            q_loss = F.mse_loss(current_q_a, target_q)
        else:
            # (ANNEALING THE BIAS) important sampling
            q_loss = F.mse_loss(current_q_a, target_q, reduction='none')
            q_loss = torch.mean(weight * q_loss)
        return q_loss, td_error
    
    def _train_standard(self, s, a, r, s_prime, dw):
        '''The standard Deep Q-Learning'''
        self._update_target_network()

        q_loss, _ = self._dqn_loss(s, a, r, s_prime, dw)
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

    def train_minimax_group_by_sampling(self, replay_buffer: ReplayBuffer, n: int):
        self._update_target_network()

        grouped_grads = []
        grouped_loss = []
        for i in range(n):
            s, a, r, s_prime, dw = \
                self._transition_to_device(*replay_buffer.sample(self.batch_size))
            q_loss, _ = self._dqn_loss(s, a, r, s_prime, dw)
            self.optimizer.zero_grad()
            q_loss.backward()
            grads = grads_to_vector(self.q_net.parameters())

            grouped_grads.append(grads.cpu().detach())
            grouped_loss.append(q_loss.cpu().detach())

        grouped_loss = torch.stack(grouped_loss)
        # q_loss = torch.mean(grouped_loss)
        grouped_grads = torch.stack(grouped_grads)

        grads = grad_by_minimax(grouped_grads, grouped_loss)

        vector_to_grads(
            torch.tensor(grads, dtype=torch.float).to(self.device), 
            self.q_net.parameters()
        )

        self.optimizer.step()
        self.training_step += 1

    def train_with_proportional_per(self, replay_buffer: ProportionalPrioritizedReplayBuffer, beta: float):
        self._update_target_network()

        samples = replay_buffer.sample(self.batch_size, beta)
        s, a, r, s_prime, dw = self._transition_to_device(*samples['transitions'])

        q_loss, td_error = self._dqn_loss(
            s, a, r, s_prime, dw, torch.tensor(samples['weights'], dtype=torch.float).to(self.device)
        )

        # Calculate priorities for replay buffer $p_i = |\delta_i| + \epsilon$
        new_priorities = np.abs(td_error.cpu().detach().numpy()) + 1e-6
        replay_buffer.update_priorities(samples['indexes'], new_priorities)

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        self.training_step += 1

    def train_minimax_proportional_per(self, replay_buffer: ProportionalPrioritizedReplayBuffer, beta: float):
        pass

    def save(self, model_path: str):
        torch.save(self.q_net.state_dict(), model_path)

    def load(self, model_path: str):
        self.q_net.load_state_dict(torch.load(model_path))