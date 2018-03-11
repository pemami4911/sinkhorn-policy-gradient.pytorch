#!/usr/bin/env python

import argparse
import os
from tqdm import tqdm 
from collections import namedtuple
import json
import pprint as pp
import numpy as np
import pdb

import torch
import torch.optim as optim
import torch.autograd as autograd
# import torch.multiprocessing as _mp
# mp = _mp.get_context('spawn')
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value, Logger

from spg.models import SPGMLPActor, SPGSiameseActor
import spg.util as util

from envs import dataset

from sklearn.utils.linear_assignment_ import linear_assignment

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser(description="")

# Data
parser.add_argument('--task', default='sort_0-24')
parser.add_argument('--arch', type=str, default='siamese')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--parallel_envs', type=int, default=128)
parser.add_argument('--train_size', type=int, default=1000000)
parser.add_argument('--val_size', type=int, default=1000)
parser.add_argument('--test_size', type=int, default=10000)
# Model cfg options here
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--n_features', type=int, default=2)
parser.add_argument('--n_nodes', type=int, default=50)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--lstm_dim', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--sinkhorn_iters', type=int, default=5)
parser.add_argument('--sinkhorn_tau', type=float, default=1)
parser.add_argument('--actor_lr', type=float, default=5e-4)
parser.add_argument('--actor_lr_decay_rate', type=float, default=0.96)
parser.add_argument('--actor_lr_decay_step', type=int, default=5000)
parser.add_argument('--use_batchnorm', type=util.str2bool, default=False)
# Training cfg options here
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--random_seed', type=int, default=24601)
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--use_cuda', type=util.str2bool, default=True, help='')
# Misc
parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--save_model', type=util.str2bool, default=False, help='Save after epoch')
parser.add_argument('--actor_load_path', type=str, default='')
parser.add_argument('--critic_load_path', type=str, default='')
parser.add_argument('--disable_tensorboard', type=util.str2bool, default=False)
parser.add_argument('--disable_progress_bar', type=util.str2bool, default=False)
parser.add_argument('--sigopt', type=util.str2bool, default=False)
parser.add_argument('--_id', type=str, default='123456789')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--make_only', type=int, default=3, help='0-train,1-val,2-test,-1-all,3,none')
parser.add_argument('--use_val', type=util.str2bool, default=False)
parser.add_argument('--save_stats', type=util.str2bool, default=True)
DEBUG = False

#########################################
##          Training funcs             ##
######################################### 

def compute_loss(criterion, pred, labels):
    pred = pred.view(-1, pred.size()[-1])
    labels = labels.view(-1)
    return criterion(pred, labels)

def main(args, idx=0, suggestions=None):

    """
    Train policy with SL
    """
    args['model'] = 'sl'
    task = args['task'].split('_')
    args['COP'] = task[0]  # the combinatorial optimization problem

    if args['save_stats']:
        fglab_results_dir = os.path.join('results', 'fglab', 'sl', args['COP'], args['_id'])
        try:
            os.makedirs(fglab_results_dir)
        except:
            pass
        args['fglab_results'] = open(os.path.join(fglab_results_dir, 'scores.json'), 'w')

    if args['COP'] == 'sort':
        criteria = train_sort(args)
    elif args['COP'] == 'mwm' or args['COP'] == 'mwm2D':
        criteria = train_mwm(args)
    return criteria

def train_mwm(args):
    args['sl'] = True
    args, env, training_dataloader, validation_dataloader, test_dataloader = dataset.build(args, 0)
    # Load the model parameters from a saved state
    if args['actor_load_path'] != '' and args['critic_load_path'] != '':
        print('  [*] Loading model from {}'.format(args['load_path']))
        actor = torch.load(
            os.path.join(os.getcwd(),
                args['actor_load_path']))
        if args['use_cuda']:
            actor.cuda_after_load()
    else:
        actor = SPGSiameseActor(args['n_features'], args['n_nodes'], args['embedding_dim'], args['lstm_dim'],
                    args['sinkhorn_iters'], args['sinkhorn_tau'], args['alpha'], args['use_cuda'])
    args['save_dir'] = os.path.join(os.getcwd(), args['output_dir'],
        args['task'],
        args['run_name'])    
    try:
        os.makedirs(save_dir)
    except:
        pass
    if args['use_cuda']:
        actor = actor.cuda()
    # Optimizers
    actor_optim = optim.Adam(actor.parameters(), lr=args['actor_lr'])
    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
        range(args['actor_lr_decay_step'], args['actor_lr_decay_step'] * 1000,
            args['actor_lr_decay_step']), gamma=args['actor_lr_decay_rate'])
    epoch = args['epoch_start']
    step = int(epoch * np.ceil(args['train_size'] / float(args['batch_size'])))
    eval_step = step
    tot_R = 0. 
    eval_R = 0.
    scores = {'_scores': {}}
    x_ent_loss = torch.nn.CrossEntropyLoss()
    num_corrects = []
    eval_means = []
    if args['use_val']:
        ds = validation_dataloader
    else:
        ds = training_dataloader
    for i in range(epoch, epoch + args['n_epochs']):
        actor.train()
        for obs in tqdm(ds, disable=args['disable_progress_bar']):
            x = obs['x']
            matching = obs['matching'].long()
            x = Variable(x, requires_grad=False)
            matching = Variable(matching, requires_grad=False)
            if args['use_cuda']:
                x = x.cuda()
                matching = matching.cuda()
            proj_M, perms, _, dist = actor(x)
            # Use X-ent loss btwn proj_M and matching
            actor_loss = compute_loss(x_ent_loss, proj_M, matching) 
            actor_optim.zero_grad()
            actor_loss.backward()
            # clip gradient norms
            torch.nn.utils.clip_grad_norm(actor.parameters(),
                args['max_grad_norm'], norm_type=2)
            actor_optim.step()
            actor_scheduler.step()
            if step % args['log_step'] == 0 and not DEBUG:
                #print('step: {}, avg reward: {}'.format(
                #    step, R.mean().data[0])) 
                print('step: {}, x-ent loss: {}, dist: {}'.format(
                    step, actor_loss.data[0], dist.data[0]))
            if not args['disable_tensorboard']:
                #log_value('Reward', R.data[0], step)
                #log_value('Running avg reward', (tot_R / (step+1)).data[0], step)
                log_value('x-ent loss', actor_loss.data[0], step)
                log_value('Closeness to nearest vertex of Birkhoff Poly', dist.data[0], step)            
            step += 1
        # Eval 
        actor.eval()
        eval_R = []
        optimal_R = []
        ratios = []
        num_correct = 0
        test_loss = []
        for obs in tqdm(test_dataloader, disable=args['disable_progress_bar']):    
            x = obs['x']
            matching = obs['matching'].long()
            optimal_weight = obs['weight']
            x = Variable(x, requires_grad=False)
            optimal_weight = Variable(optimal_weight, requires_grad=False)
            matching = Variable(matching, requires_grad=False)
            if args['use_cuda']:
                x = x.cuda()
                optimal_weight = optimal_weight.cuda()
                matching = matching.cuda()
            proj_M, perms, _, dist = actor(x)
            # count the number perfectly sorted
            p = torch.zeros(args['batch_size'], args['n_nodes'], args['n_nodes'])
            for j in range(args['batch_size']):
                for k in range(args['n_nodes']):
                    p[j, k, int(matching[j,k].data[0])] = 1
            p = Variable(p, requires_grad=False)
            if args['use_cuda']:
                p = p.cuda()
            pp = torch.split(perms, 1, 0)
            qq = torch.split(p, 1, 0)
            which_correct = list(map(lambda x,y: torch.equal(x.data,y.data), pp, qq))
            num_correct += sum(which_correct)
            actor_loss = compute_loss(x_ent_loss, proj_M, matching) 
            test_loss.append(actor_loss.data[0])
            # Compute Reward
            matchings = torch.matmul(torch.transpose(x[:,args['n_nodes']:2*args['n_nodes'],:], 1, 2), perms)
            matchings = torch.transpose(matchings, 1, 2)
            matchings = torch.cat([x[:,0:args['n_nodes'],:], matchings], dim=1)
            R = env(matchings, args['use_cuda'])
            #eval_R += R.mean()
            eval_R.append(R.data.cpu().numpy())
            optimal_R.append(optimal_weight.data.cpu().numpy())
            ratios.append((R / optimal_weight.float().unsqueeze(1)).data.cpu().numpy())
        eval_step += 1
        print('eval: {}, test_loss: {}, n_correct: {}, ratio: {}, avg optimal reward: {}, avg reward: {}, min reward: {}, max_reward: {}'.format(
            eval_step, np.mean(test_loss), num_correct, np.mean(ratios), np.mean(optimal_R), np.mean(eval_R), np.min(eval_R), np.max(eval_R)))
        if not args['disable_tensorboard']:
            log_value('N_correct', num_correct, eval_step)
            log_value('avg eval reward', np.mean(eval_R), eval_step)
        num_corrects.append(num_correct)
        eval_means.append(np.mean(eval_R))
        # if args['save_model']:
        #     print(' [*] saving model...')
        #     torch.save(actor, os.path.join(args['save_dir'], 'actor-epoch-{}.pt'.format(i)))
        if args['save_stats']:
            scores['_scores']['n_correct'] = int(np.max(num_corrects))
            scores['_scores']['avg_eval_reward_{}'.format(eval_step)] = float(np.mean(eval_R))
            scores['_scores']['optimality_ratio_{}'.format(eval_step)] = float(np.mean(ratios))
    
    if args['save_stats']:
        # write training stats to file
        json.dump(scores, args['fglab_results'])
        # close files
        args['fglab_results'].close()
    return float(np.max(eval_means))

def train_sort(args):
    # Task specific configuration - generate dataset if needed
    args, env, training_dataloader, validation_dataloader = dataset.build(args)
    
    labels = Variable(torch.arange(args['sort_low'], args['sort_high'] + 1). \
        unsqueeze(0).unsqueeze(0).repeat(args['batch_size'], 1, 1)).detach()

    # Load the model parameters from a saved state
    if args['actor_load_path'] != '' and args['critic_load_path'] != '':
        print('  [*] Loading model from {}'.format(args['load_path']))

        actor = torch.load(
            os.path.join(os.getcwd(),
                args['actor_load_path']))
        if args['use_cuda']:
            actor.cuda_after_load()
    else:
            actor = SPGMLPActor(args['n_features'], args['n_nodes'], args['hidden_dim'],
                args['sinkhorn_iters'], args['sinkhorn_tau'], args['alpha'],
                args['use_cuda'], args['use_batchnorm'])
        
    args['save_dir'] = os.path.join(os.getcwd(), args['output_dir'],
        args['task'],
        args['run_name'])    
    try:
        os.makedirs(save_dir)
    except:
        pass
    if args['use_cuda']:
        actor = actor.cuda()
    # Optimizers
    actor_optim = optim.Adam(actor.parameters(), lr=args['actor_lr'])
    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
        range(args['actor_lr_decay_step'], args['actor_lr_decay_step'] * 1000,
            args['actor_lr_decay_step']), gamma=args['actor_lr_decay_rate'])
    epoch = args['epoch_start']
    step = int(epoch * np.ceil(args['train_size'] / float(args['batch_size'])))
    eval_step = step
    tot_R = 0. 
    eval_R = 0.
    reconstruction_loss = torch.nn.MSELoss()
    if args['use_cuda']:
        reconstruction_loss = reconstruction_loss.cuda()
        labels = labels.cuda()
    for i in range(epoch, epoch + args['n_epochs']):
        for obs in tqdm(training_dataloader, disable=args['disable_progress_bar']):
            obs = Variable(obs, requires_grad=False)
            if args['use_cuda']:
                obs = obs.cuda()
            proj_M, perms, X, dist = actor(obs)
            # apply the permutation to the input
            # obs^T is [N, 1, len]
            X_bar = torch.matmul(torch.transpose(obs, 1, 2), X)
            actor_loss = reconstruction_loss(X_bar, labels) 
            actor_optim.zero_grad()
            actor_loss.backward()
            # clip gradient norms
            torch.nn.utils.clip_grad_norm(actor.parameters(),
                args['max_grad_norm'], norm_type=2)
            actor_optim.step()
            actor_scheduler.step()

            if step % args['log_step'] == 0 and not DEBUG:
                #print('step: {}, avg reward: {}'.format(
                #    step, R.mean().data[0])) 
                print('step: {}, reconstruction loss: {}'.format(
                    step, actor_loss.data[0]))
                inn = []
                out = []
                for n,m in zip(torch.t(obs[0]).data[0], X_bar[0].data[0]):
                    inn.append(round(n))
                    out.append(round(m, ndigits=3))
                print('step: {}, {}'.format(step, inn))                    
                print('step: {}, {}'.format(step, out))
            if not args['disable_tensorboard']:
                log_value('Reconstruction error', actor_loss.data[0], step)
                log_value('Closeness to nearest vertex of Birkhoff Poly', dist.data[0] / args['input_dim'], step)
            step += 1
        # Eval 
        eval_R = []
        for obs in tqdm(validation_dataloader, disable=args['disable_progress_bar']):
            obs = Variable(obs, requires_grad=False)
            if args['use_cuda']:
                obs = obs.cuda()
            _, action, _, dist = actor(obs)
            # apply the permutation to the input
            solutions = torch.matmul(torch.transpose(obs, 1, 2), action)
            # Rewards are already negated in the environment for policy optimization updates
            R = env(solutions, args['use_cuda'])
            eval_R.append(R.data.cpu().numpy())
        eval_step += 1
        if not args['disable_tensorboard']:
            #log_value('Reward', R.data[0], step)
            log_value('avg eval reward', np.mean(eval_R), eval_step)
            log_value('Closeness to nearest vertex of Birkhoff Poly during eval', dist.data[0] / args['input_dim'], eval_step)

        #print('eval: {}, running avg eval reward: {}, latest avg reward: {}'.format(i,
        #    (eval_R / eval_step).data[0], (latest_R / (args['val_size'] / args['batch_size'])).data[0]))
        print('eval: {}, avg reward: {}, min reward: {}, max_reward: {}'.format(eval_step, np.mean(eval_R), np.min(eval_R), np.max(eval_R)))

    # if args['save_model']:
    #     print(' [*] saving model...')
    #     torch.save(actor, os.path.join(args['save_dir'], 'actor-epoch-{}.pt'.format(i)))
if __name__ == '__main__':
    args = vars(parser.parse_args())
    # Pretty print the run args
    pp.pprint(args)
    # Set the random seed
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    if not args['disable_tensorboard']:
        configure(os.path.join('results', 'logs', args['task'], args['run_name']))
    main(args)
