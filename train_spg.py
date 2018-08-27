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
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_size', type=int, default=500000)
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
parser.add_argument('--actor_lr', type=float, default=3e-4)
parser.add_argument('--critic_lr', type=float, default=3e-4)
parser.add_argument('--actor_lr_decay_rate', type=float, default=0.95)
parser.add_argument('--critic_lr_decay_rate', type=float, default=0.95)
parser.add_argument('--actor_lr_decay_step', type=int, default=50000)
parser.add_argument('--critic_lr_decay_step', type=int, default=5000)
parser.add_argument('--k_exchange', type=int, default=2)
parser.add_argument('--epsilon', type=float, default=1.)
parser.add_argument('--epsilon_decay_rate', type=float, default=0.97)
parser.add_argument('--epsilon_decay_step', type=int, default=500000)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--rnn_dim', type=int, default=128)
parser.add_argument('--bidirectional', type=spg_utils.str2bool, default=True)
parser.add_argument('--entropy_coeff', type=float, default=0)

# Training cfg options here
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=1234)
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--buffer_size', type=int, default=1e6)
parser.add_argument('--log_step', type=int, default=100, help='Log info every log_step steps')
parser.add_argument('--disable_critic_aux_loss', type=spg_utils.str2bool, default=False)
# TODO: remove this
parser.add_argument('--actor_workers', type=int, default=0)
# CUDA
parser.add_argument('--use_cuda', type=spg_utils.str2bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)
# Store the replay buffer on the GPU? For N <= 20 
parser.add_argument('--replay_buffer_gpu', type=spg_utils.str2bool, default=True)
# Misc
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--base_dir', type=str, default='/media/pemami/DATA/sinkhorn-pg/')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--save_model', type=spg_utils.str2bool, default=False, help='Save after epoch')
parser.add_argument('--save_stats', type=spg_utils.str2bool, default=True)
parser.add_argument('--load_actor', action='append', type=str)
parser.add_argument('--load_critic', action='append', type=str)
parser.add_argument('--disable_tensorboard', type=spg_utils.str2bool, default=True)
parser.add_argument('--disable_progress_bar', type=spg_utils.str2bool, default=False)
parser.add_argument('--_id', type=str, default='123456789', help='FGLab experiment ID')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--make_datasets', type=str, default='None')


Experience = namedtuple('Experience', ['state', 'action', 'reward'])
DEBUG = False

def train_model(args):
    # Pretty print the run args
    pp.pprint(args)

    if not args['disable_tensorboard']:
        # append last 6 digits of experiment id to run name
        args['run_name'] = args['_id'][-6:] + '-' + args['run_name']
        configure(os.path.join(args['base_dir'], 'results', 'logs', args['task'], 'train', args['run_name']), flush_secs=2)
    
    task = args['task'].split('_')
    args['COP'] = task[0]  # the combinatorial optimization problem
    
    # Load the model parameters from a saved state
    if args['load_actor'] and args['load_critic']:
        print('  [*] Loading models from {}'.format(args['load_actor']))
        actor = torch.load(
            os.path.join(args['base_dir'],
                args['load_actor']), map_location=lambda storage, loc: storage)
        critic = torch.load(
            os.path.join(args['base_dir'],
                args['load_critic']), map_location=lambda storage, loc: storage)
        if args['use_cuda']:
            actor.cuda_after_load()
            critic.cuda_after_load()
        actor.n_nodes = args['n_nodes']
        critic.n_nodes = args['n_nodes']
    else:
        # initialize RL model
        if args['arch'] == 'sequential':
            actor = SPGSequentialActor(args['parallel_envs'], args['n_features'], args['n_nodes'],
                    args['max_n_nodes'], args['embedding_dim'], args['rnn_dim'], args['sinkhorn_iters'],
                    args['sinkhorn_tau'], args['actor_workers'], args['use_cuda'])
            critic = SPGSequentialCritic(args['parallel_envs'], args['n_features'], args['n_nodes'],
                    args['max_n_nodes'], args['embedding_dim'], args['rnn_dim'], args['use_cuda'])
        elif args['arch'] == 'matching':
            actor = SPGMatchingActorV2(args['parallel_envs'], args['n_features'], args['n_nodes'],
                    args['max_n_nodes'], args['embedding_dim'],
                    args['rnn_dim'], args['annealing_iters'], args['sinkhorn_iters'],  args['sinkhorn_tau'], 
                    args['tau_decay'], args['actor_workers'], args['use_cuda'])
            critic = SPGMatchingCriticV2(args['parallel_envs'], args['n_features'], args['n_nodes'],
                    args['max_n_nodes'], args['embedding_dim'], args['rnn_dim'], args['use_cuda'])
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
    
    # LR schedules
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
    observation_shape = [args['n_nodes'], args['n_features']]
    if args['COP'] == 'mwm2D': 
        observation_shape[0] *= 2
    replay_buffer = ReplayBuffer(args['buffer_size'], action_shape=[args['n_nodes'], args['n_nodes']], 
            observation_shape=observation_shape, use_cuda=args['replay_buffer_gpu'])
    
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
    
    def eval(eval_step):
        with torch.no_grad():
            # Eval 
            eval_R = []
            eval_birkhoff_dist = []
            ratios = []
            actor.eval()
            critic.eval()
            for obs in tqdm(test_dataloader, disable=args['disable_progress_bar']):            
                soft_perm, hard_perm = actor(obs)
                dist = spg_utils.birkhoff_distance(soft_perm, hard_perm)
                
                # apply the permutation
                if args['COP'] == 'sort' or args['COP'] == 'tsp':
                    permuted_input = actor.permute_input(spg_utils.permute_sequence, hard_perm)
                    if args['COP'] == 'tsp':
                        permuted_input = torch.transpose(permuted_input, 1, 2)
                elif args['COP'] == 'mwm2D':
                    permuted_input = actor.permute_input(spg_utils.permute_bipartite, hard_perm)
                
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
            scores['_scores']['eval_avg_reward_{}'.format(train_step * args['parallel_envs'])] = mean_eval_R.item()
            #scores['_scores']['eval_dist_to_nearest_vertex_{}'.format(train_step * args['parallel_envs'])] = mean_eval_birkhoff_dist.item()
            eval_means.append(mean_eval_R.item())
            eval_stddevs.append(stddev_eval_R.item())
            if args['COP'] == 'mwm2D':
                scores['_scores']['optimality_ratio_{}'.format(train_step * args['parallel_envs'])] = float(np.mean(ratios))
            if args['COP'] == 'mwm2D':
                print('avg. optimal matching weight: {:.4f}, ratio: {}'.format(mwm2D_opt, np.mean(ratios)))
            print('eval after {} train steps, got avg reward: {:.4f} and dist to nearest vertex of Birkhoff poly: {:.4f}'.format(
               train_step * args['parallel_envs'], mean_eval_R, mean_eval_birkhoff_dist))
            if not args['disable_tensorboard']:
                log_value('Eval avg reward', mean_eval_R, eval_step)
                log_value('Eval std reward', stddev_eval_R, eval_step)
                log_value('Eval dist to nearest vertex of Birkhoff poly', mean_eval_birkhoff_dist, eval_step)
            return eval_step

    i = 0
    for i in range(epoch, epoch + args['n_epochs']):
        # We want to do the first eval on the untrained agent
        eval_step = eval(eval_step)
        if args['save_model']:
            print(' [*] saving actor and critic...')
            torch.save(actor, os.path.join(args['save_dir'], 'actor-epoch-{}.pt'.format(i+1)))
            torch.save(critic, os.path.join(args['save_dir'], 'critic-epoch-{}.pt'.format(i+1)))  
        actor.train()
        critic.train()
        for obs in tqdm(training_dataloader, disable=args['disable_progress_bar']):
            soft_perm, hard_perm = actor(obs)
            dist = spg_utils.birkhoff_distance(soft_perm, hard_perm)
            
            # Vanishing gradients in Sinkhorn layer caused the network weights to become NaN
            if hard_perm is None:
                if args['save_stats']:   
                    scores['_scores']['eval_avg_reward_{}'.format(train_step * args['parallel_envs'])] = -1
                    json.dump(scores, fglab_results)
                    fglab_results.close()
                return 0, 0
            
            # do epsilon greedy exploration
            if np.random.rand() < epsilon:
                # Add noise in the form of 2-exchange neighborhoods
                soft_perm, hard_perm = spg_utils.k_exchange(args['k_exchange'], soft_perm, hard_perm)
            if train_step > 0 and epsilon > 0.01:
                epsilon += epsilon_decay
            
            # apply the permutation
            if args['COP'] == 'sort' or args['COP'] == 'tsp':
                permuted_input = actor.permute_input(spg_utils.permute_sequence, hard_perm)
                if args['COP'] == 'tsp':
                    permuted_input = torch.transpose(permuted_input, 1, 2)
            elif args['COP'] == 'mwm2D':
                permuted_input = actor.permute_input(spg_utils.permute_bipartite, hard_perm)
            
            # compute the reward
            R = env(permuted_input, args['use_cuda'])
            
            running_avg_R.append(copy.copy(R.data.cpu().numpy()))
            running_avg_bd.append(copy.copy(dist.data.cpu().numpy()))
            if args['save_stats']: 
                tot_R.append(R.data.cpu().numpy())
                birkhoff_dist.append(dist.data.cpu().numpy())
            if train_step % args['log_step'] == 0 and not DEBUG:
                print('epoch: {}, step: {}, avg reward: {:.4f}, std dev: {:.4f}, min reward: {:.4f}, ' \
                        'max reward: {:.4f}, epsilon: {:.4f}, bd: {:.4f}, avg round time: {:.4f}'.format(
                    i+1, train_step, np.mean(running_avg_R), np.std(running_avg_R), np.min(running_avg_R),
                        np.max(running_avg_R), epsilon, np.mean(running_avg_bd), actor.total_round_time / actor.count))
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
                log_value('Exploration $\epsilon$', epsilon, train_step)
            
            if args['replay_buffer_gpu']:
                replay_buffer.append(obs.data, hard_perm.data.byte(), soft_perm.data, R.data)
            else:
                replay_buffer.append(obs.data.cpu(), hard_perm.data.byte().cpu(), soft_perm.data.cpu(), R.data.cpu())
            # sample from replay buffer if possible
            if replay_buffer.nb_entries > args['batch_size']:
                s_batch, a_batch, psi_batch, targets = replay_buffer.sample(args['batch_size'])
                a_batch = a_batch.float()
                if not args['replay_buffer_gpu'] and args['use_cuda']:                
                    s_batch.pin_memory()
                    psi_batch.pin_memory()
                    a_batch.pin_memory()
                    targets.pin_memory()
                    s_batch = Variable(s_batch.cuda(async=True))
                    psi_batch = Variable(psi_batch.cuda(async=True))
                    a_batch = Variable(a_batch.cuda(async=True))
                    targets = Variable(targets.cuda(async=True))
                else:
                    s_batch = Variable(s_batch)
                    psi_batch = Variable(psi_batch)
                    a_batch = Variable(a_batch)
                    targets = Variable(targets)
                # Compute Q(s_t, mu(s_t)=a_t)
                # size is [batch_size, 1]
                # N.B. We use the actions from the replay buffer to update the critic
                # a_batch_t are the hard permutations
                hard_Q = critic(s_batch, a_batch).squeeze(2)
                critic_out = critic_loss(hard_Q, targets)
                if not args['disable_critic_aux_loss']:
                    soft_Q = critic(s_batch, psi_batch).squeeze(2)
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
                soft_action, _ = actor(s_batch, forward_pass=False)
                #soft_action = actor(s_batch, forward_pass=False)
                # N.B. we use the action just computed from the actor net here, which 
                # will be used to compute the actor gradients
                # compute gradient of critic network w.r.t. actions, grad Q_a(s,a)
                soft_critic_out = critic(s_batch, soft_action).squeeze(2).mean()
                # Compute the consistency regularization term
                hpq = spg_utils.entropy(soft_action, soft_action)
                #actor_loss = -soft_critic_out + (args['entropy_coeff'] * hpq)
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
                    log_value('hpq', hpq.data[0], train_step)
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
    
    # Set random seeds
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    
    with torch.cuda.device(args['cuda_device']):
        print("Score: {}".format(train_model(args)))
    
