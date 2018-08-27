import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from pathos.multiprocessing import ProcessingPool as Pool
from spg.layers import Sinkhorn, DeterministicAnnealing, SetPooling
import spg.spg_utils as spg_utils

class SPGSequentialActor(nn.Module):
    """
    Embeds the input, then an RNN maps it to an intermediate representation
    which gets transofrmed to a stochastic matrix

    """
    def __init__(self, bsz, n_features, n_nodes, max_n_nodes, embedding_dim, rnn_dim,
            sinkhorn_iters=5, sinkhorn_tau=1, num_workers=4, cuda=True):
        super(SPGSequentialActor, self).__init__()
        self.use_cuda = cuda
        self.batch_size = bsz
        self.n_nodes = n_nodes
        self.max_n_nodes = max_n_nodes
        self.rnn_dim = rnn_dim
        self.num_workers = num_workers
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.pooling1 = SetPooling(embedding_dim, rnn_dim)
        self.pooling2 = SetPooling(rnn_dim, rnn_dim)
        self.pooling3 = SetPooling(rnn_dim, self.max_n_nodes)
        self.sinkhorn = Sinkhorn(sinkhorn_iters, sinkhorn_tau)
        self._input = Variable(torch.zeros(bsz, n_nodes, n_features))
        self._input.data.pin_memory()
        self._X = Variable(torch.zeros(bsz, n_nodes, n_nodes))
        self._perms = Variable(torch.zeros(bsz, n_nodes, n_nodes))
        self._perms.data.pin_memory()
        self.total_round_time = 0.
        self.count = 0.
        if num_workers > 0:
            self.pool = Pool(num_workers)
        if cuda:
            self._input = self._input.cuda(async=True)
            self._perms = self._perms.cuda(async=True)

    def cuda_after_load(self):
        self._input = self._input.cuda(async=True)
        self._perms = self._perms.cuda(async=True)

    def permute_input(self, f, hard_perm):
        return f(self._input, hard_perm)

    def net(self):
        # result is [batch_size, n_nodes, embedding_dim]
        x = F.leaky_relu(self.embedding(self._input))
        # result x is [batch_size, n_nodes, rnn_dim]
        x = self.pooling1(x)
        # result is [batch_size, n_nodes, max_n_nodes]
        x = self.pooling2(x)
        x = self.pooling3(x)
        # extract original dims by masking
        x = x[:, :, 0:self.n_nodes]
        return self.sinkhorn(x)

    def forward(self, x, forward_pass=True):
        """
        x is [batch_size, 2 * n_nodes, n_features]
        """
        self._input.data.copy_(x)
        psi = self.net()
        if forward_pass:
            start = time.time()
            self._X.data.copy_(psi.data.cpu())
            batch = self._X.numpy()
            if np.any(np.isnan(batch)):
                return None, None, None, None
            if self.num_workers > 0:
                batches = np.split(batch, self.num_workers, 0)
                perms = self.pool.map(spg_utils.matching, batches)
                perms = [p for pp in perms for p in pp]
            else:
                perms = spg_utils.matching(batch)
            self._perms.data.copy_(torch.stack(perms).contiguous())
            time_diff = time.time() - start
            self.count += 1
            self.total_round_time += time_diff
            return psi, self._perms
        else: # During the backward pass
            """
            # randomly sample a permutation
            perm = torch.zeros(self.n_nodes, self.n_nodes)
            p2 = torch.randperm(self.n_nodes, requires_grad=False).unsqueeze(1)
            idxs = torch.cat([torch.arange(0,self.n_nodes).unsqueeze(1), p2], dim=1)
            perm[idxs[:,0], idxs[:,1]] = 1
            perm = perm.unsqueeze(0).repeat(self.batch_size, 1, 1)
            if self.use_cuda: perm = perm.cuda()
            self._input.data.copy_(self.permute_input(spg_utils.permute_bipartite, perm))
            psi2 = self.net()
            # apply perm_inv
            psi2 = torch.matmul(psi2, torch.transpose(perm, 1, 2))
            """
            return psi, None

class SPGMatchingActorV2(nn.Module):
    def __init__(self, bsz, n_features, n_nodes, max_n_nodes, embedding_dim, rnn_dim,
            annealing_iters=4, sinkhorn_iters=5, sinkhorn_tau=1., tau_decay=0.8, 
            num_workers=4, cuda=True):
        super(SPGMatchingActorV2, self).__init__()
        self.use_cuda = cuda
        self.batch_size = bsz
        self.n_nodes = n_nodes
        self.max_n_nodes = max_n_nodes
        self.rnn_dim = rnn_dim
        self.num_workers = num_workers
        self.embedding = nn.Linear(n_features, embedding_dim)
        #self.gru = nn.GRU(max_n_nodes, rnn_dim)
        self.embedding2 = nn.Linear(self.max_n_nodes, rnn_dim)
        self.pooling1 = SetPooling(rnn_dim, rnn_dim)
        self.pooling2 = SetPooling(rnn_dim, rnn_dim)
        self.pooling3 = SetPooling(rnn_dim, self.max_n_nodes)
        #self.fc1 = nn.Linear(self.rnn_dim, max_n_nodes)
        #self.anneal = DeterministicAnnealing(annealing_iters,
        #        sinkhorn_iters, sinkhorn_tau, tau_decay)
        self.sinkhorn = Sinkhorn(sinkhorn_iters, sinkhorn_tau)
        self._init_hx = Variable(torch.zeros(1, self.rnn_dim))
        self._input = Variable(torch.zeros(bsz, 2 * n_nodes, n_features))
        self._input.data.pin_memory()
        self._X = Variable(torch.zeros(bsz, n_nodes, n_nodes))
        self._perms = Variable(torch.zeros(bsz, n_nodes, n_nodes))
        self._perms.data.pin_memory()
        self.total_round_time = 0.
        self.count = 0.
        if num_workers > 0:
            self.pool = Pool(num_workers)
        if cuda:
            self._init_hx = self._init_hx.cuda()
            self._input = self._input.cuda(async=True)
            self._perms = self._perms.cuda(async=True)

    def cuda_after_load(self):
        self._init_hx = self._init_hx.cuda()
        self._input = self._input.cuda(async=True)
        self._perms = self._perms.cuda(async=True)

    def permute_input(self, f, hard_perm):
        return f(self._input, hard_perm)

    def net(self):
        g1 = self._input[:,0 : self.n_nodes,:]
        g2 = self._input[:,self.n_nodes : 2*self.n_nodes,:]
        g1 = F.leaky_relu(self.embedding(g1))
        g2 = F.leaky_relu(self.embedding(g2))
        # take outer product, result is [batch_size, N, N]
        x = torch.bmm(g2, torch.transpose(g1, 2, 1))
        
        # Pad to be [batch_size, n_nodes, self.max_n_nodes]
        x = x.unsqueeze(1)
        x = F.pad(x, (0, self.max_n_nodes - self.n_nodes, 0, 0)).squeeze(1)
        x = F.leaky_relu(self.embedding2(x))

        #x = torch.transpose(x, 0, 1)
        #init_hx = self._init_hx.unsqueeze(1).repeat(1, self.batch_size, 1)
        #x, _ = self.gru(x, init_hx)
        # x is [n_nodes, batch_size, rnn_dim]
        #x = torch.transpose(x, 0, 1)
        #x = self.fc1(x)

        # result x is [batch_size, n_nodes, rnn_dim]
        x = self.pooling1(x)
        # result is [batch_size, n_nodes, max_n_nodes]
        x = self.pooling2(x)
        x = self.pooling3(x)
        # extract original dims by masking
        x = x[:, :, 0:self.n_nodes]
        return self.sinkhorn(x)

    def forward(self, x, forward_pass=True):
        """
        x is [batch_size, 2 * n_nodes, n_features]
        """
        self._input.data.copy_(x)
        psi = self.net()
        if forward_pass:
            start = time.time()
            self._X.data.copy_(psi.data.cpu())
            batch = self._X.numpy()
            if np.any(np.isnan(batch)):
                return None, None, None, None
            if self.num_workers > 0:
                batches = np.split(batch, self.num_workers, 0)
                perms = self.pool.map(spg_utils.matching, batches)
                perms = [p for pp in perms for p in pp]
            else:
                perms = spg_utils.matching(batch)
            self._perms.data.copy_(torch.stack(perms).contiguous())
            time_diff = time.time() - start
            self.count += 1
            self.total_round_time += time_diff
            return psi, self._perms
        else: # During the backward pass
            """
            # randomly sample a permutation
            perm = torch.zeros(self.n_nodes, self.n_nodes)
            p2 = torch.randperm(self.n_nodes, requires_grad=False).unsqueeze(1)
            idxs = torch.cat([torch.arange(0,self.n_nodes).unsqueeze(1), p2], dim=1)
            perm[idxs[:,0], idxs[:,1]] = 1
            perm = perm.unsqueeze(0).repeat(self.batch_size, 1, 1)
            if self.use_cuda: perm = perm.cuda()
            self._input.data.copy_(self.permute_input(spg_utils.permute_bipartite, perm))
            psi2 = self.net()
            # apply perm_inv
            psi2 = torch.matmul(psi2, torch.transpose(perm, 1, 2))
            """
            return psi, None

class SPGMatchingActor(nn.Module):
    from sklearn.utils.linear_assignment_ import linear_assignment

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
        self.sinkhorn = Sinkhorn(sinkhorn_iters, sinkhorn_tau)
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
        # g1,g2 are [batch_size, n_nodes, num_features]
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
                perms = self.pool.map(spg_utils.parallel_matching, batches)
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
            return psi, perms
        else:
            return psi, None

SPGSiameseActor = SPGMatchingActor

class SPGSequentialCritic(nn.Module):
    def __init__(self, bsz, n_features, n_nodes, max_n_nodes, embedding_dim,
            rnn_dim, cuda=True):
        super(SPGSequentialCritic, self).__init__()
        self.use_cuda = cuda
        self.batch_size = bsz
        self.n_nodes = n_nodes
        self.max_n_nodes = max_n_nodes
        self.rnn_dim = rnn_dim
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.embed_action = nn.Linear(max_n_nodes, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(max_n_nodes)
        self.pooling1 = SetPooling(embedding_dim, rnn_dim)
        self.pooling2 = SetPooling(rnn_dim, rnn_dim)
        self.pooling3 = SetPooling(rnn_dim, embedding_dim)
        #self.gru = nn.GRU(max_n_nodes, rnn_dim)
        self.combine = nn.Linear(embedding_dim, max_n_nodes)
        #self.bn1 = nn.BatchNorm1d(max_n_nodes)
        self.bn2 = nn.BatchNorm1d(max_n_nodes)
        #self.fc1 = nn.Linear(self.rnn_dim, embedding_dim)
        self.fc2 = nn.Linear(max_n_nodes, 1)
        self.fc3 = nn.Linear(max_n_nodes, 1)
        self._byte_mask = Variable(torch.zeros(1, max_n_nodes, max_n_nodes).byte())
        self._input = Variable(torch.zeros(bsz, n_nodes, n_features))
        self._input.data.pin_memory()
        if cuda:
            self._byte_mask = self._byte_mask.cuda()
            self._input = self._input.cuda(async=True)

    def cuda_after_load(self):
        self._byte_mask = self._byte_mask.cuda()
        self._input = self._input.cuda(async=True)

    def forward(self, x, p):
        """
        x is [batch_size,  n_nodes, num_features]
        p is [batch_size, n_nodes, n_nodes]
        """
        self._input.data.copy_(x)
        x = F.leaky_relu(self.embedding(self._input))
        # result is [batch_size, n_nodes, embedding_dim]
        x = self.pooling1(x)
        # result is [batch_size, n_nodes, max_n_nodes]
        x = self.pooling2(x)
        x = self.pooling3(x)
        # pad with zeros
        x = x.unsqueeze(1)
        x = F.pad(x, (0, 0, 0, self.max_n_nodes - self.n_nodes)).squeeze(1)
        
        #  pad the permutation to be [batch_size, max_n_nodes, max_n_nodes]
        #  [P 0]
        #  [0 0]
        p = p.unsqueeze(1)
        p = F.pad(p, (0, self.max_n_nodes - self.n_nodes, 0,
            self.max_n_nodes - self.n_nodes)).squeeze(1)
        p = F.leaky_relu(self.embedding_bn(self.embed_action(p)))
        # [batch_size, max_n_nodes, max_n_nodes]
        x = F.leaky_relu(self.bn2(self.combine(x + p)))
        if self.n_nodes < self.max_n_nodes:
            # mask out activations for x
            self._byte_mask[:, self.n_nodes:self.max_n_nodes,
                    self.n_nodes:self.max_n_nodes] = 1
            # repeat across 0 and 1 dims
            byte_mask = self._byte_mask.repeat(self.batch_size, 1, 1)
            x[byte_mask.detach()] = 0.
            out = torch.transpose(self.fc2(x), 1, 2)
            # out is [batch_size, 1, max_n_nodes]
            # mask out the extra dims
            out[byte_mask[:,0,:].unsqueeze(1).detach()] = 0.
        else:
            out = torch.transpose(self.fc2(x), 1, 2)
        out = self.fc3(out)
        # out is [batch_size, 1, 1]
        return out

class SPGMatchingCriticV2(nn.Module):
    def __init__(self, bsz, n_features, n_nodes, max_n_nodes,
            embedding_dim, rnn_dim, cuda):
        super(SPGMatchingCriticV2, self).__init__()
        self.use_cuda = cuda
        self.batch_size = bsz
        self.n_nodes = n_nodes
        self.max_n_nodes = max_n_nodes
        self.rnn_dim = rnn_dim
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.embedding2 = nn.Linear(max_n_nodes, rnn_dim)
        self.embed_action = nn.Linear(max_n_nodes, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(max_n_nodes)
        self.pooling1 = SetPooling(rnn_dim, rnn_dim)
        self.pooling2 = SetPooling(rnn_dim, rnn_dim)
        self.pooling3 = SetPooling(rnn_dim, embedding_dim)
        #self.gru = nn.GRU(max_n_nodes, rnn_dim)
        self.combine = nn.Linear(embedding_dim, max_n_nodes)
        self.bn1 = nn.BatchNorm1d(max_n_nodes)
        self.bn2 = nn.BatchNorm1d(max_n_nodes)
        #self.fc1 = nn.Linear(self.rnn_dim, embedding_dim)
        self.fc2 = nn.Linear(max_n_nodes, 1)
        self.fc3 = nn.Linear(max_n_nodes, 1)
        self._byte_mask = Variable(torch.zeros(1, max_n_nodes, max_n_nodes).byte())
        self._init_hx = Variable(torch.zeros(1, self.rnn_dim))
        self._input = Variable(torch.zeros(bsz, 2 * n_nodes, n_features))
        self._input.data.pin_memory()
        if cuda:
            self._init_hx = self._init_hx.cuda()
            self._byte_mask = self._byte_mask.cuda()
            self._input = self._input.cuda(async=True)

    def cuda_after_load(self):
        self._init_hx = self._init_hx.cuda()
        self._byte_mask = self._byte_mask.cuda()
        self._input = self._input.cuda(async=True)

    def forward(self, x, p):
        """
        x is [batch_size, 2 * n_nodes, num_features]
        p is [batch_size, n_nodes, n_nodes]
        """
        self._input.data.copy_(x)
        # split x into G1 and G2
        g1 = self._input[:,0 : self.n_nodes,:]
        g2 = self._input[:, self.n_nodes : 2 * self.n_nodes,:]
        g1 = F.leaky_relu(self.embedding(g1))
        g2 = F.leaky_relu(self.embedding(g2))
        # take outer product, result is [batch_size, N, N]
        x = torch.bmm(g2, torch.transpose(g1, 2, 1))
        
        # [batch_size, n_nodes, self.max_n_nodes]
        x = x.unsqueeze(1)
        x = F.pad(x, (0, self.max_n_nodes - self.n_nodes, 0, 0)).squeeze(1)

        #x = torch.transpose(x, 0, 1)
        #init_hx = self._init_hx.unsqueeze(1).repeat(1, self.batch_size, 1)
        #x, _ = self.gru(x, init_hx)
        # h is [n_nodes, batch_size, rnn_dim]
        #x = torch.transpose(x, 0, 1)
        x = F.leaky_relu(self.embedding2(x))
        # result is [batch_size, n_nodes, embedding_dim]
        x = self.pooling1(x)
        # result is [batch_size, n_nodes, max_n_nodes]
        x = self.pooling2(x)
        x = self.pooling3(x)
        # pad with zeros
        x = x.unsqueeze(1)
        x = F.pad(x, (0, 0, 0, self.max_n_nodes - self.n_nodes)).squeeze(1)
        #x = F.leaky_relu(self.bn1(self.fc1(x)))
        
        #  pad the permutation to be [batch_size, max_n_nodes, max_n_nodes]
        #  [P 0]
        #  [0 0]
        p = p.unsqueeze(1)
        p = F.pad(p, (0, self.max_n_nodes - self.n_nodes, 0,
            self.max_n_nodes - self.n_nodes)).squeeze(1)
        p = F.leaky_relu(self.embedding_bn(self.embed_action(p)))
        # [batch_size, max_n_nodes, max_n_nodes]
        x = F.leaky_relu(self.bn2(self.combine(x + p)))
        if self.n_nodes < self.max_n_nodes:
            # mask out activations for x
            self._byte_mask[:, self.n_nodes:self.max_n_nodes,
                    self.n_nodes:self.max_n_nodes] = 1
            # repeat across 0 and 1 dims
            byte_mask = self._byte_mask.repeat(self.batch_size, 1, 1)
            x[byte_mask.detach()] = 0.
            out = torch.transpose(self.fc2(x), 1, 2)
            # out is [batch_size, 1, max_n_nodes]
            # mask out the extra dims
            out[byte_mask[:,0,:].unsqueeze(1).detach()] = 0.
        else:
            out = torch.transpose(self.fc2(x), 1, 2)
        out = self.fc3(out)
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

