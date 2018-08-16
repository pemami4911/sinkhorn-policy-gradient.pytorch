#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm 
from collections import namedtuple, deque
import pprint as pp
import numpy as np
import time
import json
import h5py 
import copy
# DEBUG
import pdb

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value, Logger
from spg.models import SPGSequentialActor, SPGMatchingActor, SPGMatchingActorV2
from spg.models import SPGSequentialCritic, SPGMatchingCritic, SPGMatchingCriticV2
from spg.memory import Memory as ReplayBuffer
import spg.spg_utils as spg_utils
# tasks
from envs import dataset

parser = argparse.ArgumentParser(description="")

# Data
parser.add_argument('--task', default='tsp_10', help='Supported: {sort, mwm, mwm2D, tsp}')
parser.add_argument('--parallel_envs', type=int, default=32)
parser.add_argument('--train_size', type=int, default=5000000)
parser.add_argument('--test_size', type=int, default=10000)
# Model cfg options here
parser.add_argument('--n_features', type=int, default=2)
parser.add_argument('--n_nodes', type=int, default=10)
parser.add_argument('--max_n_nodes', type=int, default=50)
parser.add_argument('--arch', type=str, default='rnn')
parser.add_argument('--sinkhorn_iters', type=int, default=10)
parser.add_argument('--sinkhorn_tau', type=float, default=0.05)
parser.add_argument('--annealing_iters', type=int, default=4)
parser.add_argument('--tau_decay', type=float, default=0.8)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--rnn_dim', type=int, default=128)
parser.add_argument('--bidirectional', type=spg_utils.str2bool, default=True)
# Testing cfg options here
parser.add_argument('--n_samples', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=1234)
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--log_step', type=int, default=100, help='Log info every log_step steps')
parser.add_argument('--actor_workers', type=int, default=4)
# CUDA
parser.add_argument('--use_cuda', type=spg_utils.str2bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)
# Misc
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--base_dir', type=str, default='/media/pemami/DATA/sinkhorn-pg/')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--save_model', type=spg_utils.str2bool, default=False, help='Save after epoch')
parser.add_argument('--save_stats', type=spg_utils.str2bool, default=True)
parser.add_argument('--load_actor', action='append', type=str)
parser.add_argument('--disable_tensorboard', type=spg_utils.str2bool, default=True)
parser.add_argument('--disable_progress_bar', type=spg_utils.str2bool, default=False)
parser.add_argument('--_id', type=str, default='123456789', help='FGLab experiment ID')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--make_only', type=int, default=3)


def eval_model(args):
    # Pretty print the run args
    pp.pprint(args)

    if not args['disable_tensorboard']:
        # append last 6 digits of experiment id to run name
        args['run_name'] = args['_id'][-6:] + '-' + args['run_name']
        configure(os.path.join(args['base_dir'], 'results', 'logs', args['task'], 'eval', args['run_name']), flush_secs=2)
    
    task = args['task'].split('_')
    args['COP'] = task[0]  # the combinatorial optimization problem
    
    # Load the model parameters from a saved state
    print('  [*] Loading models from {}'.format(args['load_actor']))
    actor = torch.load(args['load_actor'][0], map_location=lambda storage, loc: storage)
    actor.batch_size = args['parallel_envs']
    actor.n_features = args['n_features']
    actor.n_nodes = args['n_nodes']
    actor.reinit()

    if args['use_cuda']:
        actor = actor.cuda()
        actor.cuda_after_load()

    # Get dataloaders for train and test datasets
    args, env, _, test_dataloader = dataset.build(args, args['epoch_start'])
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
    
    epoch = args['epoch_start']
    # approx, since we throw away minibatches that aren't complete
    num_steps_per_epoch = np.ceil(args['train_size'] / float(args['parallel_envs']))
    eval_step = int(epoch * (np.ceil(args['test_size'] / float(args['parallel_envs']))))
    
    scores = {'_scores': {}}
    eval_means = []
    eval_stddevs = []
    eval_ratios = []
    eval_ratios_std = []

    def eval(eval_step):
        with torch.no_grad():
            # Eval 
            eval_R = []
            eval_birkhoff_dist = []
            ratios = []
            actor.eval()
            for obs in tqdm(test_dataloader, disable=args['disable_progress_bar']):            
                soft_perm, hard_perm, _, _ = actor(obs)
                dist = spg_utils.birkhoff_distance(soft_perm, hard_perm)
                
                # apply the permutation
                if args['COP'] == 'sort' or args['COP'] == 'tsp':
                    permuted_input = actor.permute_input(spg_utils.permute_sequence, obs, hard_perm)
                    if args['COP'] == 'tsp':
                        permuted_input = torch.transpose(permuted_input, 1, 2)
                elif args['COP'] == 'mwm2D':
                    permuted_input = actor.permute_input(spg_utils.permute_bipartite, obs, hard_perm)
                
                # compute the reward
                R = env(permuted_input, args['use_cuda'])
                eval_R.append(R.data.cpu().numpy())
                eval_birkhoff_dist.append(dist.data.cpu().numpy())
                if args['COP'] == 'mwm2D':
                    ratios.append(R.data.cpu().numpy() / mwm2D_opt)
            eval_step += 1
            # flatten
            eval_R = np.array(eval_R).ravel()
            eval_birkhoff_dist = np.array(eval_birkhoff_dist).ravel()
            mean_eval_R = np.mean(eval_R)
            stddev_eval_R = np.std(eval_R)
            mean_eval_birkhoff_dist = np.mean(eval_birkhoff_dist)
            scores['_scores']['eval_avg_reward_{}'.format(eval_step)] = mean_eval_R.item()
            #scores['_scores']['eval_dist_to_nearest_vertex_{}'.format(train_step * args['parallel_envs'])] = mean_eval_birkhoff_dist.item()
            eval_means.append(mean_eval_R.item())
            eval_stddevs.append(stddev_eval_R.item())
            if args['COP'] == 'mwm2D':
                scores['_scores']['optimality_ratio_{}'.format(eval_step)] = float(np.mean(ratios))
                eval_ratios.append(np.mean(ratios))
                eval_ratios_std.append(np.std(ratios))
            if args['COP'] == 'mwm2D':
                print('avg. optimal matching weight: {:.4f}, ratio: {}'.format(mwm2D_opt, np.mean(ratios)))
            print('eval {}, avg reward: {:.4f} and Birkhoff distance: {:.4f}'.format(
               eval_step, mean_eval_R, mean_eval_birkhoff_dist))
            if not args['disable_tensorboard']:
                log_value('Eval avg reward', mean_eval_R, eval_step)
                log_value('Eval std reward', stddev_eval_R, eval_step)
                log_value('Eval dist to nearest vertex of Birkhoff poly', mean_eval_birkhoff_dist, eval_step)
            return eval_step
        
    for eval_step in tqdm(range(args['n_samples'])):
        # Set random seeds
        torch.manual_seed(args['random_seed'] + eval_step)
        np.random.seed(args['random_seed'] + eval_step)
        eval(eval_step)
    
    if args['save_stats']:
        # write training stats to file
        json.dump(scores, fglab_results)
        fglab_results.close()
    if args['COP'] == 'mwm2D':
        best_eval_mean = np.max(eval_ratios)
        best_eval_stddev = eval_ratios_std[np.argmax(eval_ratios)]
        avg_eval_mean = np.median(eval_ratios)
        avg_eval_std = np.mean(eval_ratios_std)
    else:
        best_eval_mean = np.max(eval_means)
        best_eval_stddev = eval_stddevs[np.argmax(eval_means)]
        avg_eval_mean = np.mean(eval_means)
        avg_eval_std = np.mean(eval_stddevs)

    return best_eval_mean, best_eval_stddev, avg_eval_mean, avg_eval_std

if __name__ == '__main__':
    
    args = vars(parser.parse_args())
    args['model'] = 'spg'
    args['sl'] = False
    
    with torch.cuda.device(args['cuda_device']):
        print("best: {}, stddev: {}, avg: {}, stddev: {}".format(*eval_model(args)))
    
