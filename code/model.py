import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''DQN'''
class QNet128(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNet128, self).__init__()
        print('net1')
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y

class QNet64(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNet64, self).__init__()
        print('net2')
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y

class SimpleConvQNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(SimpleConvQNet, self).__init__()

        # input_shape [C, H, W]
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Classifier
        conv_out_size = self._get_conv_out_size(input_shape)
        self.hidden = nn.Sequential(
            nn.Linear(conv_out_size, 512, bias=True),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(512, num_actions, bias=True)
        )

    def _get_conv_out_size(self, input_shape):
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.out(x)
        return x


'''DDPG'''
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class SimpleMLPActorDDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, a_low, a_high, hidden1=256, hidden2=128, init_w=3e-3):
        super(SimpleMLPActorDDPG, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

        self.a_low = a_low
        self.a_high = a_high
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, s):
        out = self.fc1(s)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        # mapping to action space
        out = (out + 1) * (self.a_high - self.a_low) / 2 + self.a_low
        return out

class SimpleMLPCriticDDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=128, init_w=3e-3):
        super(SimpleMLPCriticDDPG, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, s, a):
        out = self.fc1(s)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out