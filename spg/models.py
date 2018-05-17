import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from spg.layers import Sinkhorn
from spg.util import parallel_matching
from sklearn.utils.linear_assignment_ import linear_assignment
from pathos.multiprocessing import ProcessingPool as Pool

class SPGSequentialActor(nn.Module):
    """
    Embeds the input, then an RNN maps it to an intermediate representation
    which gets transofrmed to a stochastic matrix

    """
    def __init__(self, n_features, n_nodes, embedding_dim, rnn_dim, bidirectional=True,
            sinkhorn_iters=5, sinkhorn_tau=1, num_workers=4, cuda=True):
        super(SPGSequentialActor, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.rnn_dim = rnn_dim
        self.num_workers = num_workers
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_dim, bidirectional=bidirectional)
        scale = 2 if bidirectional else 1
        self.fc2 = nn.Linear(scale * self.rnn_dim, n_nodes)
        self.sinkhorn = Sinkhorn(n_nodes, sinkhorn_iters, sinkhorn_tau)
        self.round = linear_assignment
        init_hx = torch.zeros(scale, self.rnn_dim)
        if cuda:
            init_hx = init_hx.cuda()
        self.init_hx = Variable(init_hx, requires_grad=False)
        if num_workers > 0:
            self.pool = Pool(num_workers)

    def cuda_after_load(self):
        self.init_hx = self.init_hx.cuda()
    
    def forward(self, x, do_round=True):
        """
        x is [batch_size, n_nodes, num_features]
        """
        batch_size = x.size()[0]
        x = F.leaky_relu(self.embedding(x))
        x = torch.transpose(x, 0, 1)
        init_hx = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        h_last, _ = self.gru(x, init_hx)
        # h_last should be [n_nodes, batch_size, decoder_dim]
        x = torch.transpose(h_last, 0, 1)
        # transform to [batch_size, n_nodes, n_nodes]
        M = self.fc2(x)
        psi = self.sinkhorn(M)
        if do_round:
            batch = psi.data.cpu().numpy()
            if np.any(np.isnan(batch)):
                return None, None, None, None
            if self.num_workers > 0:
                batches = np.split(batch, self.num_workers, 0)
                perms = self.pool.map(parallel_matching, batches)
                perms = [p for pp in perms for p in pp]
            else:
                perms = []
                for i in range(batch_size):
                    perm = torch.zeros(self.n_nodes, self.n_nodes)                   
                    matching = self.round(-batch[i])
                    perm[matching[:,0], matching[:,1]] = 1
                    perms.append(perm)
            perms = torch.stack(perms)
            if self.use_cuda:
                perms = perms.cuda()
            #dist = torch.sum(torch.sum(psi * perms, dim=1), dim=1) / self.n_nodes
            return psi, perms
        else:
            return psi, None

class SPGMatchingActor(nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, rnn_dim,
            sinkhorn_iters=5, sinkhorn_tau=1., num_workers=4, cuda=True):
        super(SPGMatchingActor, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.rnn_dim = rnn_dim
        self.num_workers = num_workers
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.gru = nn.GRU(n_nodes, rnn_dim)
        self.fc1 = nn.Linear(self.rnn_dim, n_nodes)
        self.sinkhorn = Sinkhorn(n_nodes, sinkhorn_iters, sinkhorn_tau)
        self.round = linear_assignment
        init_hx = torch.zeros(1, self.rnn_dim)
        if cuda:
            init_hx = init_hx.cuda()
        self.init_hx = Variable(init_hx, requires_grad=False)
        if num_workers > 0:
            self.pool = Pool(num_workers) 

    def cuda_after_load(self):
        self.init_hx = self.init_hx.cuda()
    
    def forward(self, x, do_round=True):
        """
        x is [batch_size, 2 * n_nodes, num_features]
        """
        batch_size= x.size()[0]
        # split x into G1 and G2
        g1 = x[:,0:self.n_nodes,:]
        g2 = x[:,self.n_nodes:2*self.n_nodes,:]
        g1 = F.leaky_relu(self.embedding(g1))
        g2 = F.leaky_relu(self.embedding(g2))
        # take outer product, result is [batch_size, N, N]
        x = torch.bmm(g2, torch.transpose(g1, 2, 1))
        x = torch.transpose(x, 0, 1)
        init_hx = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        h, _ = self.gru(x, init_hx)
        # h is [n_nodes, batch_size, rnn_dim]
        h = torch.transpose(h, 0, 1)
        # result M is [batch_size, n_nodes, n_nodes]
        M = self.fc1(h)
        psi = self.sinkhorn(M)
        if do_round:
            batch = psi.data.cpu().numpy()
            if np.any(np.isnan(batch)):
                return None, None, None, None
            if self.num_workers > 0:
                batches = np.split(batch, self.num_workers, 0)
                perms = self.pool.map(parallel_matching, batches)
                perms = [p for pp in perms for p in pp]
            else:
                perms = []
                for i in range(batch_size):
                    perm = torch.zeros(self.n_nodes, self.n_nodes)
                    matching = self.round(-batch[i])
                    perm[matching[:,0], matching[:,1]] = 1
                    perms.append(perm)
            perms = torch.stack(perms).contiguous()
            perms.pin_memory()
            if self.use_cuda:
                perms = perms.cuda(async=True)
            #dist = torch.sum(torch.sum(psi * perms, dim=1), dim=1) / self.n_nodes
            return psi, perms
        else:
            return psi, None

SPGSiameseActor = SPGMatchingActor

class SPGSequentialCritic(nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, rnn_dim, bidirectional=True, cuda=True):
        super(SPGSequentialCritic, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.rnn_dim = rnn_dim
        self.embeddingX = nn.Linear(n_features, embedding_dim)
        self.embeddingP = nn.Linear(n_nodes, embedding_dim)
        self.combine = nn.Linear(embedding_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_dim, bidirectional=bidirectional)
        self.fc1 = nn.Linear(embedding_dim, 1)
        self.fc2 = nn.Linear(n_nodes, 1)
        scale = 2 if bidirectional else 1
        self.fc3 = nn.Linear(scale * rnn_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.bn2 = nn.BatchNorm1d(n_nodes)
        self.bn3 = nn.BatchNorm1d(n_nodes)
        init_hx = torch.zeros(scale, self.rnn_dim)
        if cuda:
            init_hx = init_hx.cuda()
        self.init_hx = Variable(init_hx, requires_grad=False)
    
    def cuda_after_load(self):
        self.init_hx = self.init_hx.cuda()

    def forward(self, x, p):
        """
        x is [batch_size, n_nodes, num_features]
        """
        batch_size = x.size()[0]
        x = F.leaky_relu(self.bn1(self.embeddingX(x)))
        p = F.leaky_relu(self.bn2(self.embeddingP(p)))
        xp = F.leaky_relu(self.bn3(self.combine(x + p)))
        x = torch.transpose(xp, 0, 1)
        init_hx = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        h_last, hidden_state = self.gru(x, init_hx)
        # h_last should be [n_nodes, batch_size, decoder_dim]
        x = torch.transpose(h_last, 0, 1)
        x = F.leaky_relu(self.fc3(x))
        out = self.fc1(x)
        out = self.fc2(torch.transpose(out, 1, 2))
        # out is [batch_size, 1, 1]
        return out

class SPGMatchingCritic(nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, rnn_dim, cuda):
        super(SPGMatchingCritic, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.rnn_dim = rnn_dim
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.embed_action = nn.Linear(n_nodes, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(n_nodes)
        self.gru = nn.GRU(n_nodes, rnn_dim)
        self.combine = nn.Linear(embedding_dim, n_nodes)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.bn2 = nn.BatchNorm1d(n_nodes)
        self.fc1 = nn.Linear(self.rnn_dim, embedding_dim)
        self.fc2 = nn.Linear(n_nodes, 1)
        self.fc3 = nn.Linear(n_nodes, 1)
        init_hx = torch.zeros(1, self.rnn_dim)
        if cuda:
            init_hx = init_hx.cuda()
        self.init_hx = Variable(init_hx, requires_grad=False)
    
    def cuda_after_load(self):
        self.init_hx = self.init_hx.cuda()

    def forward(self, x, p):
        """
        x is [batch_size, 2 * n_nodes, num_features]
        p is [batch_size, n_nodes, n_nodes]
        """
        batch_size = x.size()[0]
        # split x into G1 and G2
        g1 = x[:,0:self.n_nodes,:]
        g2 = x[:,self.n_nodes:2*self.n_nodes,:]
        g1 = F.leaky_relu(self.embedding(g1))
        g2 = F.leaky_relu(self.embedding(g2))
        # take outer product, result is [batch_size, N, N]
        x = torch.bmm(g2, torch.transpose(g1, 2, 1))
        x = torch.transpose(x, 0, 1)
        init_hx = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        h, hidden_state = self.gru(x, init_hx)
        # h is [n_nodes, batch_size, rnn_dim]
        x = torch.transpose(h, 0, 1)
        # result is [batch_size, n_nodes, embedding_dim]
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        p = F.leaky_relu(self.embedding_bn(self.embed_action(p)))
        x = F.leaky_relu(self.bn2(self.combine(x + p)))
        out = self.fc2(x)
        out = self.fc3(torch.transpose(out, 1, 2))
        # out is [batch_size, 1, 1]
        return out

