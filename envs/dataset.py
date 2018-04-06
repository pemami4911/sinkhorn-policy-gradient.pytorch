import os
from envs import sorting_task
from envs import mwm2D_task
from envs import tsp_task
from torch.utils.data import DataLoader

def build(args, epoch):
    # Task specific configuration - generate dataset if needed
    args['data_dir'] = os.path.join('data', args['COP'], args['data'])
    task = args['task'].split('_')

    if args['COP'] == 'sort':
        sort_range = task[1].split('-')
        args['sort_low'] = int(sort_range[0])
        args['sort_high'] = int(sort_range[1])
        train_fname, test_fname = sorting_task.create_dataset(
            args['train_size'],
            args['test_size'],
            args['data_dir'],
            epoch,
            low=args['sort_low'],
            high=args['sort_high'],
            random_seed=args['random_seed'])
        training_dataset = sorting_task.SortingDataset(train_fname, use_graph=False)
        test_dataset = sorting_task.SortingDataset(test_fname, use_graph=False)
        if args['model'] == 'nco':
            env = sorting_task.reward_nco
        else:
            env = sorting_task.reward_ddpg_D
    elif args['COP'] == 'mwm2D':
        N = task[1]
        if not 'val_size' in args:
            args['val_size'] = 0
        train_dir, val_dir, test_dir = mwm2D_task.create_dataset(
            args['train_size'],
            args['val_size'],
            args['test_size'],
            args['data_dir'],
            N=int(N),
            maximal=False,
            random_seed=args['random_seed'],
            sl=args['sl'],
            only=args['make_only'])
        test_dataset = mwm2D_task.MWM2DDataset(test_dir, args['test_size'], has_labels=args['sl'])           
        training_dataset = mwm2D_task.MWM2DDataset(train_dir, args['train_size'], has_labels=args['sl'])
        #if args['val_size'] > 0:
        #    val_dataset = mwm2D_task.MWM2DDataset(val_dir, args['val_size'], has_labels=args['sl'])
        if args['model'] == 'nco':
            env = mwm2D_task.reward_nco
        else:
            env = mwm2D_task.reward
    elif args['COP'] == 'tsp':
        tour_len = int(task[1])
        train_fname, test_fname = tsp_task.create_dataset(
            args['train_size'],
            args['test_size'],
            args['data_dir'],
            tour_len=tour_len,
            epoch=epoch,
            random_seed=args['random_seed'])
        training_dataset = tsp_task.TSPDataset(train_fname)
        #if not reset:
        #    val_dataset = tsp_task.TSPDataset(val_fname)
        test_dataset = tsp_task.TSPDataset(test_fname)
        if args['model'] == 'spg':
            env = tsp_task.reward_spg
        elif args['model'] == 'nco':
            env = tsp_task.reward_nco
    # Dataloaders
    training_dataloader = DataLoader(training_dataset,
         batch_size=args['parallel_envs'], shuffle=True, drop_last=True, num_workers=args['num_workers'])
    #validation_dataloader = DataLoader(val_dataset,
    #     batch_size=args['parallel_envs'], shuffle=True, drop_last=True, num_workers=args['num_workers'])
    #if args['COP'] == 'mwm2D' and args['sl']:
    test_dataloader = DataLoader(test_dataset,
         batch_size=args['parallel_envs'], shuffle=True, drop_last=True, num_workers=args['num_workers'])
    return args, env, training_dataloader, test_dataloader
    #else:
    #    return args, env, training_dataloader, validation_dataloader
    #else:
    #    return None, None, training_dataloader, None

