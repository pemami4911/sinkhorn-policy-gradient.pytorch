import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
from spg.hungarian import Hungarian
import numpy as np

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
        self.eps = 1e-7 * torch.eye(self.n_nodes)
        self.eps = Variable(self.eps, requires_grad=False)
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters
        if cuda:
            self.ones = self.ones.cuda()
            self.eps = self.eps.cuda()
        self.use_cuda = cuda

    def cuda_after_load(self):
        self.ones = self.ones.cuda()
        self.eps = self.eps.cuda()
        
    # Has to work for both +ve and -v inputs
    def exp(self, x):
        # Compute 
        mask = (x > 50).detach()
        x[mask] = torch.log(x[mask])
        return torch.exp(x)

    def row_norm(self, x):
        y = torch.matmul(torch.matmul(x, self.ones), torch.t(self.ones))
        return torch.div(x, y)

    def col_norm(self, x):
        y = torch.matmul(torch.matmul(self.ones, torch.t(self.ones)), x)
        return torch.div(x, y)

    def forward(self, x):
        #smoother = Variable(torch.from_numpy(np.random.gumbel(0, 0.1, (self.n_nodes, self.n_nodes))).float(), requires_grad=False)
        #if self.use_cuda:
        #    smoother = smoother.cuda()
        #x += smoother
        x = self.exp(x / self.tau)
        x = torch.add(x, self.eps)
        for _ in range(self.sinkhorn_iters):
            x = self.row_norm(x)
            x = self.col_norm(x)
        return x

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
    snl = Sinkhorn(n_nodes, 17, 0.1, cuda=False)

    x = Variable(torch.from_numpy(np.random.normal(size=(batch_size, n_nodes, n_nodes))).float())
    #print('input: {}'.format(x))
    out = snl.forward(x)
    print('outputs: {}, rounded: {}, dist: {}/10'.format(out, rounded_out, dist.data[0]))
    print('row sums: {} col sums: {}'.format(torch.sum(out, dim=1), torch.sum(out, dim=2)))
