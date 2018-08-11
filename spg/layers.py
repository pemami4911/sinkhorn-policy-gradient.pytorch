import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from spg.util import logsumexp

class Sinkhorn(Module):
    """
    SinkhornNorm layer from https://openreview.net/forum?id=Byt3oJ-0W
    
    If L is too large or tau is too small, gradients will disappear 
    and cause the network to NaN out!
    """    
    def __init__(self, sinkhorn_iters=5, tau=0.01):
        super(Sinkhorn, self).__init__()
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters

    def row_norm(self, x):
        """Unstable implementation"""
        #y = torch.matmul(torch.matmul(x, self.ones), torch.t(self.ones))
        #return torch.div(x, y)
        """Stable, log-scale implementation"""
        return x - logsumexp(x, dim=2, keepdim=True)

    def col_norm(self, x):
        """Unstable implementation"""
        #y = torch.matmul(torch.matmul(self.ones, torch.t(self.ones)), x)
        #return torch.div(x, y)
        """Stable, log-scale implementation"""
        return x - logsumexp(x, dim=1, keepdim=True)

    def forward(self, x, eps=1e-6):
        """ 
            x: [batch_size, N, N]
        """
        x = x / self.tau
        for _ in range(self.sinkhorn_iters):
            x = self.row_norm(x)
            x = self.col_norm(x)
        return torch.exp(x) + eps

