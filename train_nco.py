#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm 
import pprint as pp
import numpy as np
import h5py
import json
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value

from neural_combinatorial_rl.neural_combinatorial_rl import NeuralCombOptRL
from neural_combinatorial_rl.matching_nco import MatchingNeuralCombOptRL, MatchingNoDecoder
from envs import dataset

def str2bool(v):
      return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

# Data
parser.add_argument('--task', default='sort_10-20', help="The task to solve, in the form {COP}_{size}, e.g., tsp_20")
parser.add_argument('--parallel_envs', type=int, default=128, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--train_size', type=int, default=1000000, help='')
parser.add_argument('--val_size', type=int, default=10000, help='')
# Network
parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
parser.add_argument('--input_size', type=int, default=10)
parser.add_argument('--n_features', type=int, default=1)
parser.add_argument('--n_process_blocks', type=int, default=3, help='Number of process block iters to run in the Critic network')
parser.add_argument('--n_glimpses', type=int, default=2, help='No. of glimpses to use in the pointer network')
parser.add_argument('--use_tanh', type=str2bool, default=True)
parser.add_argument('--tanh_exploration', type=int, default=10, help='Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
parser.add_argument('--dropout', default=0., help='')
parser.add_argument('--terminating_symbol', default='<0>', help='')
parser.add_argument('--beam_size', default=1, help='Beam width for beam search')
# Training
parser.add_argument('--use_decoder', type=str2bool, default=False)
parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
parser.add_argument('--actor_lr_decay_step', default=5000, help='')
parser.add_argument('--critic_lr_decay_step', default=5000, help='')
parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
parser.add_argument('--reward_scale', default=2, type=float,  help='')
parser.add_argument('--is_train', type=str2bool, default=True, help='')
parser.add_argument('--n_epochs', default=1, help='')
parser.add_argument('--random_seed', type=int, default=24601, help='')
parser.add_argument('--max_grad_norm', default=1.0, help='Gradient clipping')
parser.add_argument('--use_cuda', type=str2bool, default=True, help='')
parser.add_argument('--critic_beta', type=float, default=0.9, help='Exp mvg average decay')
parser.add_argument('--use_KT', type=str2bool, default=True)
# Misc
parser.add_argument('--log_step', default=50, help='Log info every log_step steps')
parser.add_argument('--log_dir', type=str, default='results/logs')
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--base_dir', type=str, default='/data/pemami/spg/')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--disable_tensorboard', type=str2bool, default=True)
parser.add_argument('--plot_attention', type=str2bool, default=False)
parser.add_argument('--disable_progress_bar', type=str2bool, default=False)
parser.add_argument('--save_stats', type=str2bool, default=False)
parser.add_argument('--save_model', type=str2bool, default=False)
parser.add_argument('--_id', type=str, default='1234567')
parser.add_argument('--sl', type=str2bool, default=False)
parser.add_argument('--use_graph', type=str2bool, default=False)
parser.add_argument('--make_only', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--data', type=str, default='icml2018')

args = vars(parser.parse_args())
args['model'] = 'nco'
# Pretty print the run args
pp.pprint(args)
# hack
args['n_nodes'] = args['input_size']
# Set the random seed
torch.manual_seed(int(args['random_seed']))

torch.cuda.device(args['cuda_device'])

# Optionally configure tensorboard
args['run_name'] = args['_id'][-6:] + '-' + args['run_name']    
if not args['disable_tensorboard']:
    configure(os.path.join(args['base_dir'], args['log_dir'], args['task'], args['run_name']))

args['test_size'] = args['val_size']
# Task specific configuration - generate dataset if needed
task = args['task'].split('_')
args['COP'] = task[0]  # the combinatorial optimization problem
args, env, training_dataloader, test_dataloader = dataset.build(args, args['epoch_start'])
if args['COP'] == 'mwm2D':
    mwm2D_opt = test_dataloader.dataset.get_average_optimal_weight()

# Open files for writing results
if args['save_stats']:
    fglab_results_dir = os.path.join(args['base_dir'], 'results', 'fglab', args['model'], args['COP'], args['_id'])
    raw_results_dir = os.path.join(args['base_dir'], 'results', 'raw', args['model'], args['COP'], args['_id'])
    try:
        os.makedirs(fglab_results_dir)
        os.makedirs(raw_results_dir)
    except:
        pass
    fglab_results = open(os.path.join(fglab_results_dir, 'scores.json'), 'w')
    raw_results = h5py.File(os.path.join(raw_results_dir, 'raw.hdf5'), 'w')

# Load the model parameters from a saved state
if os.path.exists(args['load_path']):
    print('  [*] Loading model from {}'.format(args['load_path']))

    model = torch.load(
        os.path.join(
            os.getcwd(),
            args['load_path']
        ))
    #model.actor_net.decoder.max_length = args['input_size']
    model.is_train = args['is_train']
else:
    if args['COP'] == 'mwm2D':
        if args['use_decoder']:
            model = MatchingNeuralCombOptRL(
                args['input_size'],
                args['n_features'],
                int(args['embedding_dim']),
                int(args['hidden_dim']),
                args['input_size'], # decoder len
                args['terminating_symbol'],
                int(args['n_glimpses']),
                int(args['n_process_blocks']), 
                float(args['tanh_exploration']),
                args['use_tanh'],
                int(args['beam_size']),
                args['is_train'],
                args['use_cuda'])
        else:
            model = MatchingNoDecoder(
                    args['input_size'],
                    args['n_features'],
                    args['embedding_dim'],
                    args['hidden_dim'],
                    args['use_cuda'])
            model.mask_logits = True
    else:
        # Instantiate the Neural Combinatorial Opt with RL module
        model = NeuralCombOptRL(
            args['n_features'],
            int(args['embedding_dim']),
            int(args['hidden_dim']),
            args['input_size'], # decoder len
            args['terminating_symbol'],
            int(args['n_glimpses']),
            int(args['n_process_blocks']), 
            float(args['tanh_exploration']),
            args['use_tanh'],
            int(args['beam_size']),
            args['is_train'],
            args['use_cuda'])

args['save_dir'] = os.path.join(args['base_dir'], 'results', 'models', args['model'], args['COP'], args['_id'])    
try:
    os.makedirs(args['save_dir'])
except:
    pass

#critic_mse = torch.nn.MSELoss()
#critic_optim = optim.Adam(model.critic_net.parameters(), lr=float(args['critic_net_lr']))
actor_optim = optim.Adam(model.parameters(), lr=float(args['actor_net_lr']))
actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
        range(int(args['actor_lr_decay_step']), int(args['actor_lr_decay_step']) * 1000,
            int(args['actor_lr_decay_step'])), gamma=float(args['actor_lr_decay_rate']))

#critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
#        range(int(args['critic_lr_decay_step']), int(args['critic_lr_decay_step']) * 1000,
#            int(args['critic_lr_decay_step'])), gamma=float(args['critic_lr_decay_rate']))

critic_exp_mvg_avg = torch.zeros(1)
beta = args['critic_beta']

if args['use_cuda']:
    model = model.cuda()
    #critic_mse = critic_mse.cuda()
    critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()
step = 0
val_step = 0
tot_R = []
scores = {'_scores': {}}
if not args['is_train']:
    args['n_epochs'] = '1'
epoch = int(args['epoch_start'])

def eval(val_step, final=False):
    # Use (greedy) beam search decoding for validation
    #model.actor_net.decoder.decode_type = "greedy"
    model.decode_type("greedy")
    print('\nstarting eval\n')
    example_input = []
    example_output = []
    eval_R = []
    ratios = []
    optimal = []
    # put in test mode!
    model.eval()
    for batch_id, obs in enumerate(tqdm(test_dataloader,
            disable=args['disable_progress_bar'])):
        obs = Variable(obs, requires_grad=False)
        if args['use_cuda']:
            obs = obs.cuda()
        obs = torch.transpose(obs, 2, 1)
        probs, actions, action_idxs, _ = model(obs)
        if args['COP'] == 'sort':
            R = env(actions, args['use_KT'], args['use_cuda'])
        elif args['COP'] == 'mwm2D':
            # actions is list of len N of (batch_size, n_features)
            x1 = obs[:, :, 0:args['input_size']]
            x2 = torch.stack(actions, 2)
            a = torch.cat([x1, x2], dim=2)
            R = env(torch.transpose(a, 2, 1), args['use_cuda'])
        else:
            R = env(actions, args['use_cuda'])
        eval_R.append(R.data.cpu().numpy())
        val_step += 1
        if val_step % int(args['log_step']) == 0:
            # example_output = []
            # example_input = []
            # for idx, action in enumerate(actions):
            #     if task[0] == 'tsp':
            #         example_output.append(action_idxs[idx][0].data[0])
            #     else:
            #         example_output.append(action[0].data[0])
            #     example_input.append(bat[0, :, idx].data[0])
            #print('Example test input: {}'.format(example_input))
            #print('Example test output: {}'.format(example_output))
            print('step: {}, example reward: {}'.format(val_step, R[0].data[0]))
            # if args['plot_attention']:
            #     probs = torch.cat(probs, 0)
            #     plot_attention(example_input,
            #             example_output, probs.data.cpu().numpy())
        if args['COP'] == 'mwm2D':
            ratios.append(R.data.cpu().numpy() / mwm2D_opt)
    eval_R = np.array(eval_R).ravel()
    #if args['COP'] == 'sort':
    #    # Count how many 1's in eval_R
    #    per_incorrectly_sorted = (len(eval_R) - sum(eval_R == -1.))/(len(eval_R)) * 100.
    #    print('percent incorrectly sorted: {}'.format(per_incorrectly_sorted))
    #    scores['_scores']['percent_incorrectly_sorted_{}'.format(step * args['batch_size'])] = float(per_incorrectly_sorted)        
    mean_eval_R = np.mean(eval_R)
    std_eval_R = np.std(eval_R)
    print('Validation overall avg_reward: {}'.format(mean_eval_R))
    print('Validation overall reward std: {}'.format(std_eval_R))
    if not args['disable_tensorboard']:
        log_value('eval_avg_reward', mean_eval_R, val_step)
        log_value('eval_std_reward', std_eval_R, val_step)
    scores['_scores']['eval_avg_reward_{}'.format(step * args['batch_size'])] = float(mean_eval_R)
    #scores['_scores']['eval_std_reward_{}'.format(step * args['batch_size'])] = float(std_eval_R)
    if args['COP'] == 'mwm2D':
        print('Average optimal MWM: {}'.format(mwm2D_opt))
        scores['_scores']['optimality_ratio_{}'.format(step * args['batch_size'])] = float(np.mean(ratios))
    model.decode_type("stochastic")
    return val_step 

for i in range(epoch, epoch + int(args['n_epochs'])):
    if args['is_train']:
        # eval at 0 
        val_step = eval(val_step)
        # put in train mode!
        model.train()
        # sample_batch is [batch_size x input_dim x sourceL]
        for batch_id, obs in enumerate(tqdm(training_dataloader,
                disable=args['disable_progress_bar'])):
            obs = Variable(obs, requires_grad=False)
            if args['use_cuda']:
                obs = obs.cuda()
            obs = torch.transpose(obs, 2, 1)
            probs, actions, actions_idxs, _ = model(obs)
            if args['COP'] == 'sort':
                R = env(actions, args['use_KT'], args['use_cuda'])
            elif args['COP'] == 'mwm2D':
                # actions is list of len N of (batch_size, n_features)
                x1 = obs[:, :, 0:args['input_size']]
                x2 = torch.stack(actions, 2)
                a = torch.cat([x1, x2], dim=2)
                R = env(torch.transpose(a, 2, 1), args['use_cuda'])
            else:
                R = env(actions, args['use_cuda'])
            tot_R.append(R.data.cpu().numpy())            
            if batch_id == 0:
                critic_exp_mvg_avg = R.mean()
            else:
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())
            advantage = R - critic_exp_mvg_avg
            logprobs = 0
            nll = 0
            for prob in probs: 
                # compute the sum of the log probs
                # for each tour in the batch
                logprob = torch.log(prob)
                nll += -logprob.detach()
                logprobs = logprobs + logprob
            # guard against nan
            #nll[nll != nll] = 0.
            # clamp any -inf's to 0 to throw away this tour
            #logprobs[logprobs < -1000] = 0.
            # multiply each time step by the advanrate
            reinforce = advantage.detach() * logprobs
            actor_loss = reinforce.mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            # clip gradient norms
            torch.nn.utils.clip_grad_norm(model.parameters(),
                    float(args['max_grad_norm']), norm_type=2)
            actor_optim.step()
            actor_scheduler.step()
            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()
            #critic_scheduler.step()
            #R = R.detach()
            #critic_loss = critic_mse(v.squeeze(1), R)
            #critic_optim.zero_grad()
            #critic_loss.backward()
            #torch.nn.utils.clip_grad_norm(model.critic_net.parameters(),
            #        float(args['max_grad_norm']), norm_type=2)
            #critic_optim.step()
            step += 1
            if not args['disable_tensorboard']:
                log_value('Running_avg_reward', -1 * R.mean().data[0], step)
                log_value('actor_loss', actor_loss.data[0], step)
                #log_value('critic_loss', critic_loss.data[0], step)
                log_value('critic_exp_mvg_avg', critic_exp_mvg_avg.data[0], step)
                log_value('nll', nll.mean().data[0], step)
            if step % int(args['log_step']) == 0:
                print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(
                    i, batch_id, R.mean().data[0]))
                example_output = []
                #example_input = []
                if args['COP'] == 'sort':
                    for idx, action in enumerate(actions):
                        example_output.append(round(action[0].data[0]))  # <-- ?? 
                    #if task[0] == 'tsp':
                    #    example_output.append(actions_idxs[idx][0].data[0])
                    #else:
                    #example_input.append(sample_batch[0, :, idx][0])
                    #print('Example train input: {}'.format(example_input))
                    print('Example train output: {}'.format(example_output))
        if args['save_model']:
            print(' [*] saving model...')
            torch.save(model, os.path.join(args['save_dir'], 'nco-COP-{}-N-{}-epoch-{}.pt'.format(args['COP'], args['input_size'], i)))   
# Eval one last time
val_step = eval(val_step, True)

if args['save_stats']:
    # write training stats to file
    json.dump(scores, fglab_results)
    tot_R = np.array(tot_R).ravel()
    raw_results.create_dataset('training_rewards', data=tot_R)
    # close files
    fglab_results.close()
    raw_results.close()
