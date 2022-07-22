import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional
from model import SimpleMLPActorDDPG, SimpleMLPCriticDDPG
from replay_buffer import ReplayBuffer
from utils import vector_to_grads, grads_to_vector, grad_by_minimax, ActionNoise


class DDPG_Agent:
    '''Standard Deep Deterministic Policy Gradient (DDPG) Agent'''
    def __init__(
        self, 
        actor: Union[str, nn.Module]='simple_mlp_actor',
        target_actor: Optional[nn.Module]=None,
        critic: Union[str, nn.Module]='simple_mlp_critic',
        target_critic: Optional[nn.Module]=None,
        state_dim: Optional[int]=None,
        action_dim: Optional[int]=None,
        action_low: float=-1.,
        action_high: float=1.,
        gamma: float=0.99,
        polyak: float=0.995,
        lr_actor: float=1e-4,
        lr_critic: float=1e-3,
        batch_size: int=128,
        syn_step: int=32,
        device: str='cpu') -> None:

        if isinstance(actor, str):
            if actor == 'simple_mlp_actor':
                self.actor = SimpleMLPActorDDPG(state_dim, action_dim, action_low, action_high).to(device)
                self.target_actor = SimpleMLPActorDDPG(state_dim, action_dim, action_low, action_high).to(device)
        elif isinstance(actor, nn.Module):
            self.actor = actor.to(device)
            self.target_actor = target_actor.to(device)

        if isinstance(critic, str):
            if critic == 'simple_mlp_critic':
                self.critic = SimpleMLPCriticDDPG(state_dim, action_dim).to(device)
                self.target_critic = SimpleMLPCriticDDPG(state_dim, action_dim).to(device)
        elif isinstance(critic, nn.Module):
            self.critic = critic.to(device)
            self.target_critic = target_critic.to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.device = device
        self.training_step = 0
        self.syn_step = syn_step

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    @torch.no_grad()
    def select_action(self, state, exploration_noise: Optional[ActionNoise]=None) -> float:
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.device)
        a = self.actor(state).squeeze(-1).numpy()
        if exploration_noise is not None:
            a += exploration_noise.noise()
            a = np.clip(a, self.action_low, self.action_high)
        return a

    '''Target Netowrk'''
    def _hard_update(self, target, source) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _soft_update(self, target, source) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1.0 - self.polyak))
    
    def _update_target_network(self, hard=False) -> None:
        # update the target network every fixed steps
        if self.training_step % self.syn_step == 0:
            if hard:
                self._hard_update(self.target_actor, self.actor)
                self._hard_update(self.target_critic, self.critic)
            else:
                self._soft_update(self.target_actor, self.actor)
                self._soft_update(self.target_critic, self.critic)

    '''Loss'''
    def _q_loss(self, s, a, r, s_prime, dw) -> torch.Tensor:
        # Compute the target Q value
        with torch.no_grad():
            q_prime = self.target_critic(s_prime, self.actor(s_prime)) 
            target_q = r + (1 - dw) * self.gamma * q_prime

        # Get current Q estimates
        current_q = self.critic(s, a)

        # TD error
        # td_error = target_q - current_q

        # loss
        # print(current_q)
        q_loss = F.mse_loss(current_q, target_q)
        # print('loss: ', q_loss)
        return q_loss

    def _policy_loss(self, s) -> torch.Tensor:
        # policy gradient
        q = self.critic(s, self.actor(s))
        return -q.mean()

    '''Training'''
    def _transition_to_device(self, s, a, r, s_prime, dw) -> Tuple:
        return (
            torch.tensor(s, dtype=torch.float).to(self.device),
            torch.tensor(a, dtype=torch.int64).to(self.device),
            torch.tensor(r, dtype=torch.float).to(self.device),
            torch.tensor(s_prime, dtype=torch.float).to(self.device),
            torch.tensor(dw, dtype=torch.int64).to(self.device)
        )

    def _train_standard(self, s, a, r, s_prime, dw) -> Tuple[float, float]:
        # update q (critic) network
        q_loss = self._q_loss(s, a, r, s_prime, dw)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update policy (actor) network
        # freeze q-network during the policy learning
        for p in self.critic.parameters():
            p.requires_grad = False
        
        policy_loss = self._policy_loss(s)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # syn target network
        self.training_step += 1
        self._update_target_network()

        return q_loss.item(), policy_loss.item()

    def train_standard(self, replay_buffer: ReplayBuffer) -> Tuple[float, float]:
        s, a, r, s_prime, dw = \
            self._transition_to_device(*replay_buffer.sample(self.batch_size))
        return self._train_standard(s, a, r, s_prime, dw)

    def train_minimax_group_by_sampling(self, replay_buffer: ReplayBuffer, n: int):
        # minimax method for q (critic) network
        grouped_grads = []
        grouped_loss = []
        for _ in range(n):
            s, a, r, s_prime, dw = \
                self._transition_to_device(*replay_buffer.sample(self.batch_size))
            q_loss = self._q_loss(s, a, r, s_prime, dw)
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            grads = grads_to_vector(self.critic.parameters())

            grouped_grads.append(grads.cpu().detach())
            grouped_loss.append(q_loss.cpu().detach())
        
        grouped_loss = torch.stack(grouped_loss)
        q_loss = torch.mean(grouped_loss)
        grouped_grads = torch.stack(grouped_grads)

        grads = grad_by_minimax(grouped_grads, grouped_loss)

        vector_to_grads(
            torch.tensor(grads, dtype=torch.float).to(self.device), 
            self.critic.parameters()
        )
        self.critic_optimizer.step()

        # update policy (actor) network
        # freeze q-network during the policy learning
        for p in self.critic.parameters():
            p.requires_grad = False
        
        policy_loss = self._policy_loss(s)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # syn target network
        self.training_step += 1
        self._update_target_network()

        return q_loss, policy_loss.item()