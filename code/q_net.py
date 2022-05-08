import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
