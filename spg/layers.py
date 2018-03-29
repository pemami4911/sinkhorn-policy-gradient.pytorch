import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from util import logsumexp

class Sinkhorn(Module):
    """
    SinkhornNorm layer from https://openreview.net/forum?id=Byt3oJ-0W
    
    If L is too large or tau is too small, gradients will disappear 
    and cause the network to NaN out!
    """    
    def __init__(self, n_nodes, sinkhorn_iters=5, tau=0.01, cuda=True):
        super(Sinkhorn, self).__init__()
        self.n_nodes = n_nodes
        self.ones = Variable(torch.ones(n_nodes, 1), requires_grad=False)
        self.eps = Variable(torch.FloatTensor([1e-6]), requires_grad=False)
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters
        if cuda:
            self.ones = self.ones.cuda()
            self.eps = self.eps.cuda()
        self.use_cuda = cuda

    def cuda_after_load(self):
        self.ones = self.ones.cuda()
        self.eps = self.eps.cuda()
        
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

    def forward(self, x):
        """ 
            x: [batch_size, N, N]
        """
        x = x / self.tau
        for _ in range(self.sinkhorn_iters):
            x = self.row_norm(x)
            x = self.col_norm(x)
        return torch.exp(x) + self.eps

#class GraphConvolution(Module):
#    """
#    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#    """
#
#    def __init__(self, in_features, out_features, bias=True):
#        super(GraphConvolution, self).__init__()
#        self.in_features = in_features
#        self.out_features = out_features
#        self.weight = torch.nn.Linear(in_features, out_features)
#
#   def forward(self, x, adj):
#        support = self.weight(x)
#        output = torch.bmm(adj, support)
#        return output
#
#    def __repr__(self):
#        return self.__class__.__name__ + ' (' \
#               + str(self.in_features) + ' -> ' \
#               + str(self.out_features) + ')'
#

class LayerNorm(Module):
    " Simple, slow LayerNorm "

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = Parameter(torch.ones(features))
        self.beta = Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return torch.transpose(self.gamma * torch.transpose((x - mean / (std + self.eps)), 1, 2) + self.beta, 1, 2)

if __name__ == '__main__':
    import numpy as np

    batch_size = 10
    n_nodes = 10

    torch.set_printoptions(profile="short")
    snl = Sinkhorn(n_nodes, 10, 0.1, cuda=False)

    x = Variable(torch.from_numpy(np.random.normal(size=(batch_size, n_nodes, n_nodes))).float())
    #print('input: {}'.format(x))
    out = snl.forward(x)
    print("Sinkhorn balanced matrix: {}".format(out))
    print('row sums: {} col sums: {}'.format(torch.sum(out, dim=1), torch.sum(out, dim=2)))
