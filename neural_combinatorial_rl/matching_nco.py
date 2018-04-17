import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from neural_combinatorial_rl.neural_combinatorial_rl import Encoder
#from neural_combinatorial_rl.neural_combinatorial_rl import Attention
#from neural_combinatorial_rl.neural_combinatorial_rl import Decoder

class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, in_dim, out_dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(out_dim, out_dim)
        self.project_ref = nn.Conv1d(in_dim, out_dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        v = torch.FloatTensor(out_dim)
        if use_cuda:
            v = v.cuda()
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(out_dim)) , 1. / math.sqrt(out_dim))

    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x hidden_dim x batch_size
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(2, 1, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits

class Decoder(nn.Module):
    def __init__(self, 
             embedding_dim,
             hidden_dim,
             glimpse_dim,
             max_length,
             tanh_exploration,
             terminating_symbol,
             use_tanh,
             decode_type,
             n_glimpses=1,
             beam_size=0,
             use_cuda=True):
         super(Decoder, self).__init__()
       
         self.embedding_dim = embedding_dim
         self.hidden_dim = hidden_dim
         self.n_glimpses = n_glimpses
         self.max_length = max_length
         self.terminating_symbol = terminating_symbol 
         self.decode_type = decode_type
         self.beam_size = beam_size
         self.use_cuda = use_cuda
         self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
         self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)
         self.pointer = Attention(hidden_dim, hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=self.use_cuda)
         self.glimpse = Attention(hidden_dim, hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
         self.sm = nn.Softmax()

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):    
         if mask is None:
             mask = torch.zeros(logits.size()).byte()
         if self.use_cuda: # move from GPU to CPU for speedup
             #logits = logits.cpu()
             mask = mask.cuda()
         maskk = mask.clone()
         # to prevent them from being reselected. 
         if prev_idxs is not None:
             #if self.use_cuda:
             #    prev_idxs = prev_idxs.cpu()
             # set most recently selected idx values to 1
             maskk[[x for x in range(logits.size(0))],
                     prev_idxs.data] = 1
             #for x in range(logits.size(0)):
             #    maskk[x, prev_idxs.data[x]] = 1
             logits[maskk] = -np.inf
         # move back to GPU
         #if self.use_cuda:
         #    logits = logits.cuda()
         return logits, maskk

    def forward(self, decoder_input, embedded_inputs, hidden, context):
         """
         Args:
             decoder_input: The initial input to the decoder
                 size is [batch_size x embedding_dim]. Trainable parameter.
             embedded_inputs: [sourceL x batch_size x embedding_dim]
             hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                 Initially this is set to (enc_h[-1], enc_c[-1])
             context: encoder outputs, [sourceL x batch_size x hidden_dim] 
         """
         def recurrence(x, hidden, logit_mask, prev_idxs, step):
           
             hx, cx = hidden  # batch_size x hidden_dim
           
             gates = self.input_weights(x) + self.hidden_weights(hx)
             ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
             ingate = F.sigmoid(ingate)
             forgetgate = F.sigmoid(forgetgate)
             cellgate = F.tanh(cellgate)
             outgate = F.sigmoid(outgate)
             cy = (forgetgate * cx) + (ingate * cellgate)
             hy = outgate * F.tanh(cy)  # batch_size x hidden_dim
             g_l = hy
             for i in range(self.n_glimpses):
                 ref, logits = self.glimpse(g_l, context)
                 logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
                 # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = 
                 # [batch_size x h_dim x 1]
                 g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2) 
             _, logits = self.pointer(g_l, context)
           
             logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
             probs = self.sm(logits)
             return hy, cy, probs, logit_mask
   
         batch_size = context.size(1)
         outputs = []
         selections = []
         steps = range(self.max_length)  # or until terminating symbol ?
         inps = []
         idxs = None
         mask = None
      
         if self.decode_type == "stochastic" or self.decode_type == "greedy":
             for i in steps:
                 hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
                 hidden = (hx, cx)
                 # select the next inputs for the decoder [batch_size x hidden_dim]
                 if self.decode_type == "stochastic":
                     fn = self.decode_stochastic
                 elif self.decode_type == "greedy":
                     fn = self.decode_greedy
                 decoder_input, idxs = fn(
                     probs,
                     embedded_inputs,
                     selections)
                 inps.append(decoder_input) 
                 # use outs to point to next object
                 outputs.append(probs)
                 selections.append(idxs)
             return (outputs, selections), hidden

    def decode_stochastic(self, probs, embedded_inputs, selections):
         """
         Return the next input for the decoder by sampling from the 
         input probabilities

         Args: 
             probs: [batch_size x sourceL]
             embedded_inputs: [sourceL x batch_size x embedding_dim]
             selections: list of all of the previously selected indices during decoding
        Returns:
             Tensor of size [batch_size x sourceL] containing the embeddings
             from the inputs corresponding to the [batch_size] indices
             selected for this iteration of the decoding, as well as the 
             corresponding indicies
         """
         batch_size = probs.size(0)
         # idxs is [batch_size]
         idxs = probs.multinomial().squeeze(1)

         # due to race conditions, might need to resample here
         for old_idxs in selections:
             # compare new idxs
             # elementwise with the previous idxs. If any matches,
             # then need to resample
             if old_idxs.eq(idxs).data.any():
                 print(' [!] resampling due to race condition')
                 idxs = probs.multinomial().squeeze(1)
                 break

         sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :] 
         return sels, idxs

    def decode_greedy(self, probs, embedded_inputs, selections):
         """
         Return the next input for the decoder by selecting the 
         input corresponding to the max output

         Args: 
             probs: [batch_size x sourceL]
             embedded_inputs: [sourceL x batch_size x embedding_dim]
             selections: list of all of the previously selected indices during decoding
         Returns:
             Tensor of size [batch_size x sourceL] containing the embeddings
             from the inputs corresponding to the [batch_size] indices
             selected for this iteration of the decoding, as well as the 
             corresponding indicies
         """
         batch_size = probs.size(0)
         # idxs is [batch_size]
         _, idxs = probs.max(1)
         sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :] 
         return sels, idxs

class MatchingPointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq 
    model"""
    def __init__(self,
            n_nodes,
            embedding_dim,
            hidden_dim,
            max_decoding_len,
            terminating_symbol,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            beam_size,
            use_cuda):
        super(MatchingPointerNetwork, self).__init__()

        self.encoder = Encoder(
                n_nodes,
                hidden_dim,
                use_cuda)

        self.decoder = Decoder(
                embedding_dim,
                hidden_dim,
                embedding_dim,
                max_length=max_decoding_len,
                tanh_exploration=tanh_exploration,
                use_tanh=use_tanh,
                terminating_symbol=terminating_symbol,
                decode_type="stochastic",
                n_glimpses=n_glimpses,
                beam_size=beam_size,
                use_cuda=use_cuda)

        # Trainable initial input to the decoders
        dec_in_0 = torch.FloatTensor(embedding_dim)
        if use_cuda:
            dec_in_0 = dec_in_0.cuda()
        self.decoder_in_0 = nn.Parameter(dec_in_0)
        self.decoder_in_0.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                1. / math.sqrt(embedding_dim))
            
    def forward(self, x, x2):
        """ Propagate inputs through the network
        Args: 
            x: fused embeddings [batch_size, n, n]
            x2: embedding graph 2, [batch_size, sourceL, embedding_dim]
        """
        batch_size = x.shape[0]
        x = torch.transpose(x, 0, 1)
        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)       
        encoder_cx = encoder_cx.unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)       
        enc_h, (enc_h_t_i, enc_c_t_i) = self.encoder(x, (encoder_hx, encoder_cx))
        # encoder forward pass
        #enc_hidden = [[],[]]
        #enc_h = []
        #for i in range(2):
        #    enc_h_i, (enc_h_t_i, enc_c_t_i) = self.encoder(inputs[i], (encoder_hx, encoder_cx))
        #    enc_h.append(enc_h_i)
        #    enc_hidden[0].append(enc_h_t_i[-1])
        #    enc_hidden[1].append(enc_c_t_i[-1])
        dec_init_state = (enc_h_t_i[-1], enc_c_t_i[-1])
        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(batch_size, 1)
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(
                decoder_input,
                torch.transpose(x2, 0, 1),
                dec_init_state,
                enc_h)

        return pointer_probs, input_idxs

class MatchingNoDecoder(nn.Module):
    def __init__(self, n_nodes, input_dim, embedding_dim, hidden_dim, use_cuda):
        super(MatchingNoDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.to_logits = nn.Linear(hidden_dim, n_nodes)
        self.encoder = nn.GRU(n_nodes, hidden_dim)
        self.n_nodes = n_nodes
        self.init_hx = Variable(torch.zeros(1, hidden_dim), requires_grad=False)
        self.sm = nn.LogSoftmax()
        self.decode = "stochastic"
        self.mask_logits = True
        self.use_cuda = use_cuda
        if use_cuda:
            self.init_hx = self.init_hx.cuda()
    
    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):    
         if mask is None:
             mask = torch.zeros(logits.size()).byte()
         if self.use_cuda:
            mask = mask.cuda()
         maskk = mask.clone()
         # to prevent them from being reselected. 
         if prev_idxs is not None:
             maskk[[x for x in range(logits.size(0))],
                     prev_idxs.data] = 1
             logits[maskk] = -np.inf
         return logits, maskk
    
    def decode_type(self, dt):
        self.decode = dt
    
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = torch.transpose(x[:, :, 0:self.n_nodes], 2, 1)
        x2 = torch.transpose(x[:, :, self.n_nodes:self.n_nodes*2], 2, 1)
        x1 = F.leaky_relu(self.embedding(x1))
        x2 = F.leaky_relu(self.embedding(x2))
        xx = torch.bmm(x1, torch.transpose(x2, 1, 2))
        xx = torch.transpose(xx, 0, 1)
        xx = torch.chunk(xx, self.n_nodes, 0)
        h = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        
        logit_mask = None
        idxs = None
        selections = []
        probs = []
        
        for i in range(self.n_nodes):
            enc_h, h = self.encoder(xx[i], h)
            # Need something extra here?
            logits = 10 * F.tanh(self.to_logits(enc_h.squeeze()))
            if self.mask_logits: 
                masked_logits, logit_mask = self.apply_mask_to_logits(i, logits.clone(), logit_mask, idxs)
                soft_probs = self.sm(masked_logits)
            else:
                soft_probs = self.sm(logits)
            if self.decode == "stochastic":
                # idxs is [batch_size]
                idxs = torch.exp(soft_probs).multinomial().squeeze(1)
                # due to race conditions, might need to resample here
                if self.mask_logits:
                    for old_idxs in selections:
                        # compare new idxs
                        # elementwise with the previous idxs. If any matches,
                        # then need to resample
                        if old_idxs.eq(idxs).data.any():
                            print(' [!] resampling due to race condition')
                            idxs = torch.exp(soft_probs).multinomial().squeeze(1)
                            break
            else:
                _, idxs = soft_probs.max(1)
            selections.append(idxs)
            probs.append(soft_probs)
        # Select the actions (inputs pointed to 
        # by the pointer net) and the corresponding
        # logits
        # should be size [batch_size x 
        actions = []
        # x is [batch_size, input_dim, sourceL]
        x_ = torch.transpose(x[:, :, self.n_nodes:self.n_nodes*2], 1, 2)
        # x_ is [batch_size, sourceL, input_dim]
        for action_id in selections:
            actions.append(x_[[x for x in range(batch_size)], action_id.data, :])

        #if self.is_train:
        # logits_ is a list of len sourceL of [batch_size x sourceL]
        probs_ = []
        for p, action_id in zip(probs, selections):
            probs_.append(p[[x for x in range(batch_size)], action_id.data])
        #else:
            # return the list of len sourceL of [batch_size x sourceL]
        #    probs_ = probs

        return probs_, actions, selections, torch.stack(probs)

class MatchingNeuralCombOptRL(nn.Module):
    """
    This module contains the PointerNetwork (actor) and
    CriticNetwork (critic). It requires
    an application-specific reward function
    """
    def __init__(self,
            n_nodes,
            input_dim,
            embedding_dim,
            hidden_dim,
            max_decoding_len,
            terminating_symbol,
            n_glimpses,
            n_process_block_iters,
            tanh_exploration,
            use_tanh,
            beam_size,
            is_train,
            use_cuda):
        super(MatchingNeuralCombOptRL, self).__init__()
        self.input_dim = input_dim
        self.is_train = is_train
        self.use_cuda = use_cuda
        
        self.actor_net = MatchingPointerNetwork(
                n_nodes,
                embedding_dim,
                hidden_dim,
                max_decoding_len,
                terminating_symbol,
                n_glimpses,
                tanh_exploration,
                use_tanh,
                beam_size,
                use_cuda)
       
        #embedding_ = torch.FloatTensor(input_dim,
        #    embedding_dim)
        #if self.use_cuda: 
        #    embedding_ = embedding_.cuda()
        #self.embedding = nn.Parameter(embedding_) 
        #self.embedding.data.uniform_(-(1. / math.sqrt(embedding_dim)),
        #    1. / math.sqrt(embedding_dim))
        self.embedding = nn.Linear(input_dim, embedding_dim)
    
    def decode_type(self, dt):
        self.actor_net.decoder.decode_type = dt

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_dim, 2 * sourceL]
        """
        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        sourceL = int(inputs.size(2)/2)
        x1 = torch.transpose(inputs[:, :, 0:sourceL], 2, 1)
        x2 = torch.transpose(inputs[:, :, sourceL:sourceL*2], 2, 1)
        #x = (inputs[:, :, 0 : sourceL], inputs[:, :, sourceL:sourceL * 2])
        # repeat embeddings across batch_size
        # result is [batch_size x input_dim x embedding_dim]
        #embedding = self.embedding.repeat(batch_size, 1, 1)  
        #embedded_inputs = [[], []]
        # result is [batch_size, 1, input_dim, sourceL] 
        #for i in range(2):           
        #    ips = x[i].unsqueeze(1)
        #    for j in range(sourceL):
        #        # [batch_size x 1 x input_dim] * [batch_size x input_dim x embedding_dim]
        #        # result is [batch_size, embedding_dim]
        #        embedded_inputs[i].append(torch.bmm(
        #            ips[:, :, :, j].float(),
        #            embedding).squeeze(1))
        #    # Result is [sourceL x batch_size x embedding_dim]
        #    embedded_inputs[i] = torch.cat(embedded_inputs[i]).view(
        #            sourceL,
        #            batch_size,
        #            embedding.size(2))

        # Result is [batch_size, sourceL, embedding_dim]
        x1 = F.leaky_relu(self.embedding(x1))
        x2 = F.leaky_relu(self.embedding(x2))
        # Do the outer product here
        # [batch_size, sourceL, sourceL)
        fused_embedding = torch.bmm(x1, torch.transpose(x2, 1, 2))
        # query the actor net for the input indices 
        # making up the output, and the pointer attn 
        probs_, action_idxs = self.actor_net(fused_embedding, x2)
       
        # Select the actions (inputs pointed to 
        # by the pointer net) and the corresponding
        # logits
        # should be size [batch_size x 
        actions = []
        # inputs is [batch_size, input_dim, sourceL]
        inputs_ = inputs[:, :, sourceL:sourceL * 2].transpose(1, 2)
        # inputs_ is [batch_size, sourceL, input_dim]
        for action_id in action_idxs:
            actions.append(inputs_[[x for x in range(batch_size)], action_id.data, :])

        if self.is_train:
            # probs_ is a list of len sourceL of [batch_size x sourceL]
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                probs.append(prob[[x for x in range(batch_size)], action_id.data])
        else:
            # return the list of len sourceL of [batch_size x sourceL]
            probs = probs_

        # get the critic value fn estimates for the baseline
        # [batch_size]
        #v = self.critic_net(embedded_inputs)
        # [batch_size]
        #R = self.objective_fn(actions, self.use_cuda)
        return probs, actions, action_idxs, torch.stack(probs_)
