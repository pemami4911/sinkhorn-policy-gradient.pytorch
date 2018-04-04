#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm 
from collections import namedtuple, deque
import pprint as pp
import numpy as np
import pdb
import time
import json
import h5py 
import copy
import pickle

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value, Logger
from spg.models import SPGMLPActor, SPGRNNActor, SPGSiameseActor
from spg.models import SPGMLPCritic, SPGRNNCritic, SPGSiameseCritic
from spg.replay_buffer import ReplayBuffer
import spg.util as util

# tasks
from envs import dataset

parser = argparse.ArgumentParser(description="")

# Data
parser.add_argument('--task', default='tsp_10', help='Supported: {sort, mwm, mwm2D, tsp}')
parser.add_argument('--parallel_envs', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_size', type=int, default=500000)
parser.add_argument('--test_size', type=int, default=10000)
# Model cfg options here
parser.add_argument('--n_features', type=int, default=2)
parser.add_argument('--n_nodes', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--arch', type=str, default='rnn')
parser.add_argument('--sinkhorn_iters', type=int, default=10)
parser.add_argument('--sinkhorn_tau', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--actor_lr', type=float, default=3e-4)
parser.add_argument('--critic_lr', type=float, default=3e-4)
parser.add_argument('--actor_lr_decay_rate', type=float, default=0.95)
parser.add_argument('--critic_lr_decay_rate', type=float, default=0.95)
parser.add_argument('--actor_lr_decay_step', type=int, default=50000)
parser.add_argument('--critic_lr_decay_step', type=int, default=5000)
parser.add_argument('--poisson_lambda', type=float, default=4)
parser.add_argument('--poisson_decay_rate', type=float, default=0.95)
parser.add_argument('--poisson_decay_step', type=int, default=500000)
parser.add_argument('--epsilon', type=float, default=1.)
parser.add_argument('--epsilon_decay_rate', type=float, default=0.97)
parser.add_argument('--epsilon_decay_step', type=int, default=500000)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--rnn_dim', type=int, default=128)
# Training cfg options here
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=1234)
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--buffer_size', type=int, default=1e6)
parser.add_argument('--log_step', type=int, default=100, help='Log info every log_step steps')
parser.add_argument('--disable_critic_aux_loss', type=util.str2bool, default=False)
parser.add_argument('--actor_workers', type=int, default=4)
# CUDA
parser.add_argument('--use_cuda', type=util.str2bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)
# Misc
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--base_dir', type=str, default='/media/pemami/DATA/sinkhorn-pg/')
parser.add_argument('--data', type=str, default='icml2018')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--save_model', type=util.str2bool, default=False, help='Save after epoch')
parser.add_argument('--save_stats', type=util.str2bool, default=True)
parser.add_argument('--actor_load_path', type=str, default='')
parser.add_argument('--critic_load_path', type=str, default='')
parser.add_argument('--disable_tensorboard', type=util.str2bool, default=True)
parser.add_argument('--disable_progress_bar', type=util.str2bool, default=False)
parser.add_argument('--_id', type=str, default='123456789', help='FGLab experiment ID')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--make_only', type=int, default=3)

Experience = namedtuple('Experience', ['state', 'action', 'reward'])

DEBUG = False

#########################################
##          Training funcs             ##
######################################### 

def evaluate_model(args, count):
    # Pretty print the run args
    pp.pprint(args)
 
    if not args['disable_tensorboard'] and count == 0:
        # append last 6 digits of experiment id to run name
        args['run_name'] = args['_id'][-6:] + '-' + args['run_name']
        configure(os.path.join(args['base_dir'], 'results', 'logs', args['task'], args['run_name']), flush_secs=2)
    
    task = args['task'].split('_')
    args['COP'] = task[0]  # the combinatorial optimization problem
    
    # Load the model parameters from a saved state
    if args['actor_load_path'] != '' and args['critic_load_path'] != '':
        print('  [*] Loading models from {}'.format(args['critic_load_path']))
        actor = torch.load(
            os.path.join(os.getcwd(),
                args['actor_load_path']), map_location=lambda storage, loc: storage)
        critic = torch.load(
            os.path.join(os.getcwd(),
                args['critic_load_path']), map_location=lambda storage, loc: storage)
        if args['use_cuda']:
            actor.cuda_after_load()
            critic.cuda_after_load()

    else:
        # initialize RL model
        if args['arch'] == 'fc':
            actor = SPGMLPActor(args['n_features'], args['n_nodes'], args['hidden_dim'],
                    args['use_cuda'], args['sinkhorn_iters'], args['sinkhorn_tau'], args['alpha'])
            critic = SPGMLPCritic(args['n_features'], args['n_nodes'], args['hidden_dim'])
        elif args['arch'] == 'rnn':
            actor = SPGRNNActor(args['n_features'], args['n_nodes'], args['embedding_dim'],
                    args['rnn_dim'], args['sinkhorn_iters'],
                    args['sinkhorn_tau'], args['alpha'], args['actor_workers'], args['use_cuda'])
            critic = SPGRNNCritic(args['n_features'], args['n_nodes'], args['embedding_dim'],
                    args['rnn_dim'], args['use_cuda'])
        elif args['arch'] == 'siamese':
            actor = SPGSiameseActor(args['n_features'], args['n_nodes'], args['embedding_dim'],
                args['rnn_dim'], args['sinkhorn_iters'],  args['sinkhorn_tau'], args['alpha'],
                args['actor_workers'], args['use_cuda'])
            critic = SPGSiameseCritic(args['n_features'], args['n_nodes'], args['embedding_dim'],
                args['rnn_dim'], args['use_cuda'])
    args['save_dir'] = os.path.join(args['base_dir'], 'results', 'models', args['COP'], 'spg', args['arch'], args['_id'])    
    try:
        os.makedirs(args['save_dir'])
    except:
        pass
    
    if args['use_cuda']:
        actor = actor.cuda()
        critic = critic.cuda()
    #Optimizers
    actor_optim = optim.Adam(actor.parameters(), lr=args['actor_lr'])
    critic_optim = optim.Adam(critic.parameters(), lr=args['critic_lr'])
    critic_loss = torch.nn.MSELoss()
    critic_aux_loss = torch.nn.MSELoss()

    if args['use_cuda']:
        critic_loss = critic_loss.cuda()
        critic_aux_loss = critic_aux_loss.cuda()

    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
        range(args['actor_lr_decay_step'], args['actor_lr_decay_step'] * 1000,
            args['actor_lr_decay_step']), gamma=args['actor_lr_decay_rate'])
    critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
        range(args['critic_lr_decay_step'], args['critic_lr_decay_step'] * 1000,
            args['critic_lr_decay_step']), gamma=args['critic_lr_decay_rate'])

    # Count the number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, actor.parameters())
    print("# of trainable actor parameters: {}".format(sum([np.prod(p.size()) for p in model_parameters])))
    model_parameters = filter(lambda p: p.requires_grad, critic.parameters())
    print("# of trainable critic parameters: {}".format(sum([np.prod(p.size()) for p in model_parameters])))
    # Instantiate replay buffer
    replay_buffer = ReplayBuffer(args['buffer_size'], args['random_seed'])
    # Get dataloaders for train and test datasets
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
    
    epoch = args['epoch_start']
    # approx, since we throw away minibatches that aren't complete
    num_steps_per_epoch = np.ceil(args['train_size'] / float(args['parallel_envs']))
    train_step = int(epoch * num_steps_per_epoch)
    eval_step = int(epoch * (np.ceil(args['test_size'] / float(args['parallel_envs']))))
    poisson_lambda = args['poisson_lambda'] 
    poisson_step = args['poisson_decay_step']
    poisson_decay = ((poisson_lambda * args['poisson_decay_rate']) - poisson_lambda) / (poisson_step / float(args['parallel_envs']))
    epsilon = args['epsilon']
    epsilon_step = args['epsilon_decay_step']
    epsilon_decay = ((epsilon * args['epsilon_decay_rate']) - epsilon) / (epsilon_step / float(args['parallel_envs']))
    
    running_avg_R = deque(maxlen=100)
    running_avg_bd = deque(maxlen=100)
    tot_R = []
    birkhoff_dist = []
    scores = {'_scores': {}}
    eval_means = []
    eval_stddevs = []

    def eval(eval_step, final=False):
        # Eval 
        eval_R = []
        eval_birkhoff_dist = []
        ratios = []
        actor.eval()
        critic.eval()
        for obs in tqdm(test_dataloader, disable=args['disable_progress_bar']):            
            obs = obs.pin_memory()
            obs = Variable(obs, requires_grad=False)
            if args['use_cuda']:
                obs = obs.cuda(async=True)
            _,  action, _, dist = actor(obs)
            if args['COP'] == 'sort' or args['COP'] == 'tsp':
                # apply the permutation to the input
                solutions = torch.matmul(torch.transpose(obs, 1, 2), action)
                if args['n_features'] > 1:
                    solutions = torch.transpose(solutions, 1, 2)                
                R = env(solutions, args['use_cuda'])
            elif args['COP'] == 'mwm2D':
                matchings = torch.matmul(torch.transpose(obs[:,args['n_nodes']:2*args['n_nodes'],:], 1, 2), action)
                matchings = torch.transpose(matchings, 1, 2)               
                matchings = torch.cat([obs[:,0:args['n_nodes'],:], matchings], dim=1)
                # concat result 
                R = env(matchings, args['use_cuda'])
                #if args['COP'] == 'mwm2D':
                    # compute ratio
                #    optimal_matchings.append(optimal_weight.numpy())
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
        if args['COP'] == 'mwm2D':
            print('avg. optimal matching weight: {:.4f}, ratio: {}'.format(mwm2D_opt, np.mean(ratios)))
        print('eval after {} train steps, got avg reward: {:.4f} and dist to nearest vertex of Birkhoff poly: {:.4f}'.format(
           train_step * args['parallel_envs'], mean_eval_R, mean_eval_birkhoff_dist))
        if not args['disable_tensorboard']:
            log_value('Eval avg reward', mean_eval_R, eval_step)
            log_value('Eval std reward', stddev_eval_R, eval_step)
            log_value('Eval dist to nearest vertex of Birkhoff poly', mean_eval_birkhoff_dist, eval_step)
        scores['_scores']['eval_avg_reward_{}'.format(train_step * args['parallel_envs'])] = mean_eval_R.item()
        #scores['_scores']['eval_dist_to_nearest_vertex_{}'.format(train_step * args['parallel_envs'])] = mean_eval_birkhoff_dist.item()
        if args['COP'] == 'mwm2D':
            scores['_scores']['optimality_ratio_{}'.format(train_step * args['parallel_envs'])] = float(np.mean(ratios))
        eval_means.append(mean_eval_R.item())
        eval_stddevs.append(stddev_eval_R.item())
        return eval_step

    for i in range(epoch, epoch + args['n_epochs']):
        eval_step = eval(eval_step)
        if args['save_model']:
            print(' [*] saving model...')
            torch.save(actor, os.path.join(args['save_dir'], 'actor-epoch-{}.pt'.format(i+1)))
            torch.save(critic, os.path.join(args['save_dir'], 'critic-epoch-{}.pt'.format(i+1)))  
        actor.train()
        critic.train()
        for obs in tqdm(training_dataloader, disable=args['disable_progress_bar']):
            obs.pin_memory()
            obs = Variable(obs, requires_grad=False)
            if args['use_cuda']:
                obs = obs.cuda(async=True)
            psi, action, X, dist = actor(obs)
            if action is None: # Nan'd out
                if args['save_stats']:   
                    scores['_scores']['eval_avg_reward_{}'.format(train_step * args['parallel_envs'])] = -1
                    json.dump(scores, fglab_results)
                    fglab_results.close()
                return 0, 0
            # do epsilon greedy exploration
            if np.random.rand() < epsilon:
                # Add noise in the form of 2-exchange neighborhoods
                # number of row-exchanges 
                #n_rows = np.random.poisson(poisson_lambda)
                #for idx in range(args['parallel_envs']):
                for r in range(2):
                    # randomly choose two row idxs
                    idxs = np.random.randint(0, args['n_nodes'], size=2)
                    # swap the two rows
                    tmp = action[:, idxs[0]].clone()
                    tmp2 = action[:, idxs[1]].clone()
                    tmp3 = psi[:, idxs[0]].clone()
                    tmp4 = psi[:, idxs[1]].clone()
                    action[:, idxs[0]] = tmp2
                    action[:, idxs[1]] = tmp
                    psi[:, idxs[0]] = tmp4
                    psi[:, idxs[1]] = tmp3
            if train_step > 0 and poisson_lambda > 1:
                poisson_lambda += poisson_decay
            if train_step > 0 and epsilon > 0.01:
                epsilon += epsilon_decay
            
            if args['COP'] == 'sort' or args['COP'] == 'tsp':
                # apply the permutation to the input
                solutions = torch.matmul(torch.transpose(obs, 1, 2), action)
                if args['n_features'] > 1:
                    solutions = torch.transpose(solutions, 1, 2)
                R = env(solutions, args['use_cuda'])
            elif args['COP'] == 'mwm2D':
                matchings = torch.matmul(torch.transpose(obs[:,args['n_nodes']:2*args['n_nodes'],:], 1, 2), action)
                matchings = torch.transpose(matchings, 1, 2)               
                matchings = torch.cat([obs[:,0:args['n_nodes'],:], matchings], dim=1)
                # concat result 
                R = env(matchings, args['use_cuda'])
            
            running_avg_R.append(copy.copy(R.data.cpu().numpy()))
            running_avg_bd.append(copy.copy(dist.data.cpu().numpy()))
            if args['save_stats']: 
                tot_R.append(R.data.cpu().numpy())
                birkhoff_dist.append(dist.data.cpu().numpy())
            
            if train_step % args['log_step'] == 0 and not DEBUG:
                print('epoch: {}, step: {}, avg reward: {:.4f}, std dev: {:.4f}, min reward: {:.4f}, ' \
                        'max reward: {:.4f}, poisson_lambda: {:.4f}, epsilon: {:.4f}, bd: {:.4f}'.format(
                    i+1, train_step, np.mean(running_avg_R), np.std(running_avg_R), np.min(running_avg_R),
                        np.max(running_avg_R), poisson_lambda, epsilon, np.mean(running_avg_bd))) 
                if args['COP'] == 'sort':
                    inn = []
                    out = []
                    for n,m in zip(torch.t(obs[0]).data[0], solutions[0].data[0]):
                        inn.append(n)
                        out.append(m)
                    print('step: {}, {}'.format(train_step, inn))                    
                    print('step: {}, {}'.format(train_step, out))

            if not args['disable_tensorboard']:
                log_value('Running avg reward', np.mean(running_avg_R), train_step)
                log_value('Running avg std dev', np.std(running_avg_R), train_step)
                log_value('Closeness to nearest vertex of Birkhoff Poly', np.mean(running_avg_bd), train_step)
                log_value('Action exploration Lambda', poisson_lambda, train_step)

            replay_buffer.store(obs.data, psi.data, action.data, R.data)
            
            # sample from replay buffer if possible
            if replay_buffer.size() > args['batch_size']:
                s_batch, psi_batch, a_batch, r_batch = replay_buffer.sample_batch(args['batch_size'])
                s_batch_t = Variable(torch.stack(s_batch))
                psi_batch_t = Variable(torch.stack(psi_batch))
                a_batch_t = Variable(torch.stack(a_batch)) # make sure it's disconnected from previous subgraph
                targets = Variable(torch.stack(r_batch))
                # Compute Q(s_t, mu(s_t)=a_t)
                # size is [batch_size, 1]
                # N.B. We use the actions from the replay buffer to update the critic
                # a_batch_t are the hard permutations
                
                hard_Q = critic(s_batch_t, a_batch_t).squeeze(2)
                critic_out = critic_loss(hard_Q, targets)
                if not args['disable_critic_aux_loss']:
                    soft_Q = critic(s_batch_t, psi_batch_t).squeeze(2)
                    critic_aux_out = critic_aux_loss(soft_Q, hard_Q.detach())
                    critic_optim.zero_grad()
                    (critic_out + critic_aux_out).backward()
                else:
                    critic_optim.zero_grad()
                    critic_out.backward() 
                # clip gradient norms
                torch.nn.utils.clip_grad_norm(critic.parameters(),
                    args['max_grad_norm'], norm_type=2)
                critic_optim.step()
                critic_scheduler.step()                 
                
                critic_optim.zero_grad()                
                actor_optim.zero_grad()
                soft_action, _, _, _ = actor(s_batch_t, do_round=False)
                # N.B. we use the action just computed from the actor net here, which 
                # will be used to compute the actor gradients
                # compute gradient of critic network w.r.t. actions, grad Q_a(s,a)
                soft_critic_out = critic(s_batch_t, soft_action).squeeze(2).mean()
                # penalty
                actor_loss = -soft_critic_out
                actor_loss.backward()

                # clip gradient norms
                torch.nn.utils.clip_grad_norm(actor.parameters(),
                    args['max_grad_norm'], norm_type=2)

                actor_optim.step()
                actor_scheduler.step()

                if not args['disable_tensorboard']:
                    log_value('actor loss', actor_loss.data[0], train_step)
                    log_value('critic loss', critic_out.data[0], train_step)
                    log_value('avg hard Q', hard_Q.mean().data[0], train_step)  
                    if not args['disable_critic_aux_loss']:
                        log_value('avg soft Q', soft_Q.mean().data[0], train_step)

            train_step += 1
        
    # Eval one last time
    eval_step = eval(eval_step)
    if args['save_model']:
        print(' [*] saving model...')
        torch.save(actor, os.path.join(args['save_dir'], 'actor-epoch-{}.pt'.format(i+1)))
        torch.save(critic, os.path.join(args['save_dir'], 'critic-epoch-{}.pt'.format(i+1)))  
    if args['save_stats']:
        # write training stats to file
        json.dump(scores, fglab_results)
        tot_R = np.array(tot_R).ravel()
        birkhoff_dist = np.array(birkhoff_dist).ravel()
        raw_results.create_dataset('training_rewards', data=tot_R)
        raw_results.create_dataset('birkhoff_distance', data=birkhoff_dist)
        #raw_results.create_dataset('eval_mean_rewards', data=eval_means)
        #raw_results.create_dataset('eval_stddev_rewards', data=eval_stddevs)
        # close files
        fglab_results.close()
        raw_results.close()
    best_eval_mean = np.max(eval_means)
    best_eval_stddev = eval_stddevs[np.argmax(eval_means)]
        
    return best_eval_mean, best_eval_stddev

if __name__ == '__main__':
    
    args = vars(parser.parse_args())
    args['model'] = 'spg'
    args['sl'] = False
    # Set the random seed
    torch.manual_seed(args['random_seed'])
    #torch.cuda.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    with torch.cuda.device(args['cuda_device']):
        print("Score: {}".format(evaluate_model(args, 0)))
    
