import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from spg.layers import Sinkhorn, LayerNorm
from sklearn.utils.linear_assignment_ import linear_assignment
from spg.hungarian import Hungarian

class SPGMLPActor(nn.Module):
    """
    Saw slightly slower performance using ThreadPoolExecutor. The GIL!

    """
    def __init__(self, n_features, n_nodes, hidden_dim, 
            sinkhorn_iters=5, sinkhorn_tau=1, alpha=1., cuda=True, use_batchnorm=True):
        super(SPGMLPActor, self).__init__()
        self.use_cuda = cuda
        self.use_batchnorm = use_batchnorm
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_nodes)
        self.sinkhorn = Sinkhorn(n_nodes, sinkhorn_iters, sinkhorn_tau, cuda=cuda)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.bn2 = nn.BatchNorm1d(n_nodes)
        self.round = linear_assignment

    def forward(self, x, do_round=True):
        # [N, n_nodes, hidden_dim]
        batch_size = x.size()[0]
        if self.use_batchnorm:
            x = self.bn1(self.fc1(x))
        else:
            x = self.fc1(x)
        x = F.leaky_relu(x)
        if self.use_batchnorm:
            M = self.bn2(self.fc2(x))
        else:
            M = self.fc2(x)
        psi = self.sinkhorn(M)

        if do_round:
            perms = []
            batch = psi.data.cpu().numpy()
            for i in range(batch_size):
                perm = torch.zeros(self.n_nodes, self.n_nodes)
                matching = self.round(-batch[i])
                perm[matching[:,0], matching[:,1]] = 1
                #perms.append(perm)
                #_, perm = self.round2(batch[i])
                #perm = torch.from_numpy(perm).float()
                perms.append(perm)
            perms = Variable(torch.stack(perms), requires_grad=False)
            if self.use_cuda:
                perms = perms.cuda()

            dist = torch.sum(torch.sum(psi * perms, dim=1), dim=1) / self.n_nodes
            X = ((1 - self.alpha) * perms) + self.alpha * psi 
            return psi, perms, X, dist
        else:
            return None, None, psi, None

class SPGReservoirActor(nn.Module):
    """
    Embeds the input, then an LSTM maps it to an intermediate representation
    which gets transofrmed to a stochastic matrix

    """
    def __init__(self, n_features, n_nodes, embedding_dim, lstm_dim, n_layers,
            sinkhorn_iters=5, sinkhorn_tau=1, alpha=1., cuda=True):
        super(SPGReservoirActor, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.lstm_layers = [nn.LSTM(embedding_dim, lstm_dim) for _ in range(n_layers)]
        self.fc1 = nn.Linear(self.lstm_dim, embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, n_nodes)
        self.sinkhorn = Sinkhorn(n_nodes, sinkhorn_iters, sinkhorn_tau, cuda)
        self.round = linear_assignment
        init_hx = Variable(torch.zeros(1, self.lstm_dim), requires_grad=False)
        init_cx = Variable(torch.zeros(1, self.lstm_dim), requires_grad=False)
        if cuda:
            init_hx = init_hx.cuda()
            init_cx = init_cx.cuda()
            #self.lstm[0] = self.lstm[0].cuda()
            for l in self.lstm_layers:
                l = l.cuda()
        self.init_state = (init_hx, init_cx)

    def forward(self, x, do_round=True):
        """
        x is [batch_size, n_nodes, num_features]
        """
        (batch_size, _, n_features) = x.size()
        x = F.leaky_relu(self.embedding(x))
        x = torch.transpose(x, 0, 1)
        (init_hx, init_cx) = self.init_state
        init_h = init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        init_c = init_cx.unsqueeze(1).repeat(1, batch_size, 1)
        hidden_state = (init_h, init_c)
        for lstm in self.lstm_layers:
            h_last, hidden_state = lstm(x, hidden_state)
        # h_last should be [n_nodes, batch_size, decoder_dim]
        x = torch.transpose(h_last, 0, 1)
        # transform to [batch_size, n_nodes, n_nodes]
        x = F.leaky_relu(self.fc1(x))
        M = self.fc2(x)
        psi = self.sinkhorn(M)
        if do_round:
            perms = []
            batch = psi.data.cpu().numpy()
            for i in range(batch_size):
                perm = torch.zeros(self.n_nodes, self.n_nodes)
                matching = self.round(-batch[i])
                perm[matching[:,0], matching[:,1]] = 1
                perms.append(perm)
            perms = Variable(torch.stack(perms), requires_grad=False)
            if self.use_cuda:
                perms = perms.cuda()
            dist = torch.sum(torch.sum(psi * perms, dim=1), dim=1) / self.n_nodes
            X = ((1 - self.alpha) * perms) + self.alpha * psi
            return psi, perms, X, dist
        else:
            return None, None, psi, None

class SPGSiameseActor(nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, lstm_dim,
            sinkhorn_iters=5, sinkhorn_tau=1., alpha=1., cuda=True,
            disable_lstm=False, use_layer_norm=False):
        super(SPGSiameseActor, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.lstm_dim = lstm_dim
        self.use_layer_norm = use_layer_norm
        self.alpha = alpha
        self.disable_lstm = disable_lstm
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.ln = LayerNorm(n_nodes)
        # removed embedding_bn
        self.lstm = nn.LSTM(n_nodes, lstm_dim)
        self.fc1 = nn.Linear(self.lstm_dim, n_nodes)
        self.sinkhorn = Sinkhorn(n_nodes, sinkhorn_iters, sinkhorn_tau, cuda)
        self.round = linear_assignment
        init_hx = torch.zeros(1, self.lstm_dim)
        init_cx = torch.zeros(1, self.lstm_dim)
        if cuda:
            init_hx = init_hx.cuda()
            init_cx = init_cx.cuda()
        init_hx = Variable(init_hx, requires_grad=False)
        init_cx = Variable(init_cx, requires_grad=False)
        self.init_state = (init_hx, init_cx)
        #self.round2 = Hungarian()

    def cuda_after_load(self):
        init_hx = self.init_state[0].cuda()
        init_cx = self.init_state[1].cuda()
        self.init_state = (init_hx, init_cx)
        self.sinkhorn.cuda_after_load()

    def forward(self, x, do_round=True):
        """
        x is [batch_size, 2 * n_nodes, num_features]
        """
        (batch_size, _, n_features) = x.size()
        # split x into G1 and G2
        g1 = x[:,0:self.n_nodes,:]
        g2 = x[:,self.n_nodes:2*self.n_nodes,:]
        g1 = self.embedding(g1)
        g2 = self.embedding(g2)
        if self.use_layer_norm:
            g1 = self.ln(g1)
            g2 = self.ln(g2)
        g1 = F.leaky_relu(g1)
        g2 = F.leaky_relu(g2)
        # take outer product, result is [batch_size, N, N]
        x = torch.bmm(g2, torch.transpose(g1, 2, 1))
        if not self.disable_lstm:
            x = torch.transpose(x, 0, 1)
            (init_hx, init_cx) = self.init_state
            init_h = init_hx.unsqueeze(1).repeat(1, batch_size, 1)
            init_c = init_cx.unsqueeze(1).repeat(1, batch_size, 1)
            hidden_state = (init_h, init_c)
            h, hidden_state = self.lstm(x, hidden_state)
            # h is [n_nodes, batch_size, lstm_dim]
            h = torch.transpose(h, 0, 1)
            # result M is [batch_size, n_nodes, n_nodes]
            M = self.fc1(h)
        else:
            M = x
        psi = self.sinkhorn(M)
        if do_round:
            perms = []
            batch = psi.data.cpu().numpy()
            for i in range(batch_size):
                perm = torch.zeros(self.n_nodes, self.n_nodes)
                matching = self.round(-batch[i])
                perm[matching[:,0], matching[:,1]] = 1
                #_, perm = self.round2(batch[i])
                #perm = torch.from_numpy(perm).float()
                perms.append(perm)
            perms = Variable(torch.stack(perms), requires_grad=False)
            if self.use_cuda:
                perms = perms.cuda()
            dist = torch.sum(torch.sum(psi * perms, dim=1), dim=1) / self.n_nodes
            X = ((1 - self.alpha) * perms) + self.alpha * psi
            return psi, perms, X, dist
        else:
            return None, None, psi, None

class SPGMLPCritic(nn.Module):
    def __init__(self, n_features, n_nodes, hidden_dim):
        super(SPGMLPCritic, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(n_nodes, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.combine = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.bn2 = nn.BatchNorm1d(n_nodes)
        self.bn3 = nn.BatchNorm1d(n_nodes)

        # output layer
        self.out1 = nn.Linear(hidden_dim, 1)
        self.out2 = nn.Linear(n_nodes, 1)

    def forward(self, x, p):
        # x has dim [batch, n, nhid1]
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        # p has dim [batch, n, nhid1]
        p = F.leaky_relu(self.bn2(self.fc2(p)))
        # combine x and p
        # output xp has dimension [batch, n, nhid1]
        xp = F.leaky_relu(self.bn3(self.combine(x + p)))
        # output is [batch, n, 1]
        xp = self.out1(xp)
        # output is [batch, 1], Q(s,a)
        out = self.out2(torch.transpose(xp, 2, 1))
        return out

class SPGReservoirCritic(nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, lstm_dim,
            n_layers, cuda=True):
        super(SPGReservoirCritic, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.n_layers = n_layers
        self.embeddingX = nn.Linear(n_features, embedding_dim)
        self.embeddingP = nn.Linear(n_nodes, embedding_dim)
        self.combine = nn.Linear(embedding_dim, embedding_dim)           
        self.lstm_layers = [nn.LSTM(embedding_dim, lstm_dim) for _ in range(n_layers)]
        self.fc1 = nn.Linear(lstm_dim, 1)
        self.fc2 = nn.Linear(n_nodes, 1)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.bn2 = nn.BatchNorm1d(n_nodes)
        self.bn3 = nn.BatchNorm1d(n_nodes)
        init_hx = Variable(torch.zeros(1, self.lstm_dim), requires_grad=False)
        init_cx = Variable(torch.zeros(1, self.lstm_dim), requires_grad=False)
        if cuda:
            init_hx = init_hx.cuda()
            init_cx = init_cx.cuda()
            for l in self.lstm_layers:
                l = l.cuda()
        self.init_state = (init_hx, init_cx)

    def forward(self, x, p):
        """
        x is [batch_size, n_nodes, num_features]
        """
        (batch_size, _, n_features) = x.size()
        x = F.leaky_relu(self.bn1(self.embeddingX(x)))
        p = F.leaky_relu(self.bn2(self.embeddingP(p)))
        xp = F.leaky_relu(self.bn3(self.combine(x + p)))
        xp = torch.transpose(xp, 0, 1)
        (init_hx, init_cx) = self.init_state
        init_h = init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        init_c = init_cx.unsqueeze(1).repeat(1, batch_size, 1)
        hidden_state = (init_h, init_c)
        for lstm in self.lstm_layers:
            h_last, hidden_state = lstm(xp, hidden_state)
        # h_last should be [n_nodes, batch_size, decoder_dim]
        x = torch.transpose(h_last, 0, 1)
        out = self.fc1(x)
        out = self.fc2(torch.transpose(out, 1, 2))
        # out is [batch_size, 1, 1]
        return out

class SPGSiameseCritic(nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, lstm_dim, cuda, use_layer_norm):
        super(SPGSiameseCritic, self).__init__()
        self.use_cuda = cuda
        self.n_nodes = n_nodes
        self.lstm_dim = lstm_dim
        self.use_layer_norm = use_layer_norm
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.embed_action = nn.Linear(n_nodes, embedding_dim)
        #self.embedding_bn = nn.BatchNorm1d(n_nodes)
        self.lstm = nn.LSTM(n_nodes, lstm_dim)
        self.combine = nn.Linear(embedding_dim, n_nodes)
        #self.bn1 = nn.BatchNorm1d(n_nodes)
        self.fc1 = nn.Linear(self.lstm_dim, embedding_dim)
        self.fc2 = nn.Linear(n_nodes, 1)
        self.fc3 = nn.Linear(n_nodes, 1)
        self.ln1 = LayerNorm(n_nodes)
        self.ln2 = LayerNorm(n_nodes)
        self.ln3 = LayerNorm(n_nodes)
        self.ln4 = LayerNorm(n_nodes)
        init_hx = Variable(torch.zeros(1, self.lstm_dim), requires_grad=False)
        init_cx = Variable(torch.zeros(1, self.lstm_dim), requires_grad=False)
        if cuda:
            init_hx = init_hx.cuda()
            init_cx = init_cx.cuda()
        self.init_state = (init_hx, init_cx)

    def cuda_after_load(self):
        init_hx = self.init_state[0].cuda()
        init_cx = self.init_state[1].cuda()
        self.init_state = (init_hx, init_cx)

    def forward(self, x, p):
        """
        x is [batch_size, 2 * n_nodes, num_features]
        p is [batch_size, n_nodes, n_nodes]
        """
        (batch_size, _, n_features) = x.size()
        # split x into G1 and G2
        g1 = x[:,0:self.n_nodes,:]
        g2 = x[:,self.n_nodes:2*self.n_nodes,:]
        g1 = self.embedding(g1)
        g2 = self.embedding(g2)
        if self.use_layer_norm:
            g1 = self.ln1(g1)
            g2 = self.ln1(g2)
        g1 = F.leaky_relu(g1)
        g2 = F.leaky_relu(g2)
        # take outer product, result is [batch_size, N, N]
        x = torch.bmm(g2, torch.transpose(g1, 2, 1))
        x = torch.transpose(x, 0, 1)
        (init_hx, init_cx) = self.init_state
        init_h = init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        init_c = init_cx.unsqueeze(1).repeat(1, batch_size, 1)
        hidden_state = (init_h, init_c)
        h, hidden_state = self.lstm(x, hidden_state)
        # h is [n_nodes, batch_size, lstm_dim]
        x = torch.transpose(h, 0, 1)
        # result is [batch_size, n_nodes, n_nodes]
        x = self.fc1(x)
        p = self.embed_action(p)        
        if self.use_layer_norm:
            x = self.ln2(x)
            p = self.ln3(p)            
        x = F.leaky_relu(x)
        p = F.leaky_relu(p)
        x = self.combine(x + p)
        if self.use_layer_norm: 
            x = self.ln4(x)       
        x = F.leaky_relu(x)                
        out = self.fc2(x)
        out = self.fc3(torch.transpose(out, 1, 2))
        # out is [batch_size, 1, 1]
        return out

