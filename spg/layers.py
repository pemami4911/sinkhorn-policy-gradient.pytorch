import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from spg.util import logsumexp

class Sinkhorn(Module):
    """
    Based on Sinkhorn Networks from https://arxiv.org/abs/1802.08665
    
    If L is too large or tau is too small, gradients will disappear 
    and cause the network to NaN out!
    """    
    def __init__(self, sinkhorn_iters=5, tau=0.01):
        super(Sinkhorn, self).__init__()
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters

    def row_norm(self, x):
        # Unstable implementation
        # y = torch.matmul(torch.matmul(x, self.ones), torch.t(self.ones))
        # return torch.div(x, y)
        """Stable, log-scale implementation"""
        return x - logsumexp(x, dim=2, keepdim=True)

    def col_norm(self, x):
        # Unstable implementation
        # y = torch.matmul(torch.matmul(self.ones, torch.t(self.ones)), x)
        # return torch.div(x, y)
        """Stable, log-scale implementation"""
        return x - logsumexp(x, dim=1, keepdim=True)

    def forward(self, x, eps=1e-6, tau=None):
        """ 
            x: [batch_size, N, N]
        """
        if tau is not None:
            self.tau = tau
        x = x / self.tau
        for _ in range(self.sinkhorn_iters):
            x = self.row_norm(x)
            x = self.col_norm(x)
        return torch.exp(x) + eps

class DeterministicAnnealing(Module):
    """
    As described in https://www.sciencedirect.com/science/article/pii/S0031320398800101
    for matching point sets. Sinkhorn layer is used as a subroutine; the
    main difference is that we perform the Sinkhorn operation for L iterations,
    instead of letting X converge before lowering the temperature
    """
    def __init__(self, annealing_iters, sinkhorn_iters, tau, tau_decay):
        super(DeterministicAnnealing, self).__init__()
        self.annealing_iters = annealing_iters
        self.tau = tau
        self.tau_decay = tau_decay
        self.sinkhorn = Sinkhorn(sinkhorn_iters)

    def forward(self, x):
        """ 
            x: [batch_size, N, N]
        """
        tau = self.tau
        Q = x
        for i in range(self.annealing_iters):
            Q = self.sinkhorn(Q, tau)
            if i < self.annealing_iters-1:
                # decay tau
                tau *= self.tau_decay
                # "Skip-connection" + "warm-up" for next sinkhorn
                Q = Q + 0.01 * x
        return Q

