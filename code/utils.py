import os
import numpy as np
import torch
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from cvxopt.solvers import qp
from typing import Iterable, Optional
from math import sqrt
import matplotlib.pyplot as plt
import json

'''Mini-max Method'''
def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def grads_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Convert grads of parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The grads of parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.grad.view(-1))
    return torch.cat(vec)


def vector_to_grads(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the grads of parameters

    Args:
        vec (Tensor): a single vector represents the grads of parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param

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

    res = qp(P, q, G, h, A, b)
    d = np.array(D).T.dot(np.array(res['x']))[:, 0]

    return d

# def grad_by_minimax(grouped_grads: torch.Tensor, grouped_loss: torch.Tensor):
#     r""" Calculate grads by minimax algorithm

#     Args: 
#         grouped_grads (Tensor[n, m]): grads vector of n groups
#         grouped_loss (Tensor[n, 1]): loss vector of n groups

#     Returns:

#     """

#     n, m = grouped_grads.shape

#     r"""
#     minimze (1/2)xPx + qx          |         min (1/2)xDDx - fx
#     subject to Gx <= h             |         subject to \sum_i^n x_i = 1
#                Ax = b              |                    x_i >= 0

#     D: grouped_grads, tensor of shape (n, m)
#     """

#     # D and h
#     D = matrix(grouped_grads.numpy().astype(np.float64))
#     f = matrix(grouped_loss.numpy().reshape(-1).astype(np.float64))

#     P = D * D.T
#     q = -f

#     G = matrix(-np.eye(n))
#     h = matrix(np.zeros(n))
#     A = matrix(np.ones([1, n]))
#     b = matrix(np.ones([1]))

#     res = qp(P, q, G, h, A, b)
#     d = np.array(D).T.dot(np.array(res['x']))[:, 0]

#     return d

'''Noise'''
class ActionNoise:
    def __init__(self) -> None:
        pass

    def noise(self):
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

class GaussianExplorationNoise(ActionNoise):
    def __init__(self, mu=0., sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        return np.random.normal(self.mu, self.sigma)

    def __str__(self) -> str:
        return 'NormalNoise(mu{}sigma{})'.format(self.mu, self.sigma)


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu=0., sigma=.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def __str__(self):
        return 'OUNoise(mu{}sigma{})'.format(self.mu, self.sigma)

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist

'''Save & Log'''
def save_result(scores, filename: str, title: str):
    result_json = os.path.join('..', 'result', filename + '.json')
    with open(result_json, 'w') as f:
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
    ax.set_title(title)
    plt.savefig(os.path.join('..', 'result', filename + '.png'))