import os
from envs import sorting_task
from envs import mwm2D_task
from envs import tsp_task
from torch.utils.data import DataLoader

def build(args, epoch, reset=False):
    # Task specific configuration - generate dataset if needed
    args['data_dir'] = os.path.join(args['base_dir'], 'data', args['COP'], 'icml2018')
    task = args['task'].split('_')

    if args['COP'] == 'sort':
        sort_range = task[1].split('-')
        args['sort_low'] = int(sort_range[0])
        args['sort_high'] = int(sort_range[1])
        train_fname, val_fname = sorting_task.create_dataset(
            args['train_size'],
            args['val_size'],
            args['data_dir'],
            epoch,
            low=args['sort_low'],
            high=args['sort_high'],
            train_only=reset,
            random_seed=args['random_seed'])
        training_dataset = sorting_task.SortingDataset(train_fname, use_graph=False)
        if not reset:
            val_dataset = sorting_task.SortingDataset(val_fname, use_graph=False)
        if args['model'] == 'nco':
            env = sorting_task.reward_nco
        else:
            env = sorting_task.reward_ddpg_D
    elif args['COP'] == 'mwm2D':
        N = task[1]
        if not args['sl']:
        #if args['model'] == 'spg' or args['model'] == 'nco':
        #    args['test_size'] = 0
            args['test_size'] = 0
        elif args['model'] == 'spg' or args['model'] == 'nco':
            args['test_size'] = args['val_size']
        train_dir, val_dir, test_dir = mwm2D_task.create_dataset(
            args['train_size'],
            args['val_size'],
            args['test_size'],
            args['data_dir'],
            N=int(N),
            random_seed=args['random_seed'],
            sl=args['sl'],
            only=args['make_only'])
        has_labels = False
        if args['sl']:
            has_labels = True
            test_dataset = mwm2D_task.MWM2DDataset(test_dir, args['test_size'], has_labels=True)            
        training_dataset = mwm2D_task.MWM2DDataset(train_dir, args['train_size'], has_labels=has_labels)
        if not reset:
            val_dataset = mwm2D_task.MWM2DDataset(val_dir, args['val_size'], has_labels=True)
        if args['model'] == 'nco':
            env = mwm2D_task.reward_nco
        else:
            env = mwm2D_task.reward
    elif args['COP'] == 'tsp':
        tour_len = int(task[1])
        train_fname, val_fname = tsp_task.create_dataset(
            args['train_size'],
            args['val_size'],
            args['data_dir'],
            tour_len=tour_len,
            epoch=epoch,
            reset=reset,
            random_seed=args['random_seed'])
        training_dataset = tsp_task.TSPDataset(train_fname)
        if not reset:
            val_dataset = tsp_task.TSPDataset(val_fname)
        if args['model'] == 'spg':
            env = tsp_task.reward_spg
        elif args['model'] == 'nco':
            env = tsp_task.reward_nco
    # Dataloaders
    training_dataloader = DataLoader(training_dataset,
         batch_size=args['parallel_envs'], shuffle=True, drop_last=True, num_workers=args['num_workers'])
    if not reset:
        validation_dataloader = DataLoader(val_dataset,
             batch_size=args['parallel_envs'], shuffle=True, drop_last=True, num_workers=args['num_workers'])
        if args['COP'] == 'mwm2D' and args['sl']:
            test_dataloader = DataLoader(test_dataset,
                batch_size=args['parallel_envs'], shuffle=True, drop_last=True, num_workers=args['num_workers'])
            return args, env, training_dataloader, validation_dataloader, test_dataloader
        else:
            return args, env, training_dataloader, validation_dataloader
    else:
        return None, None, training_dataloader, None

