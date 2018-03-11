# code based in part on
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# and from
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
import requests
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import os
import numpy as np
import re
import zipfile
import itertools
from collections import namedtuple


#######################################
# Reward Fns
#######################################
def reward_spg(solution, use_cuda):
    """
    Args:
        solution is a Tensor of size [batch_size, N, 2]
    Returns:
        Tensor of shape [batch_size] containing rewards
    """
    batch_size, N, _ = solution.data.shape
    tour_len = torch.zeros(batch_size, 1)

    if use_cuda:
        tour_len = tour_len.cuda()
    for i in range(N - 1):
        tour_len += torch.norm(solution[:,i,:].data - solution[:,i+1,:].data, p=2, dim=1)
    tour_len += torch.norm(solution[:,N-1,:].data - solution[:,0,:].data, p=2, dim=1)

    return Variable(-tour_len, requires_grad=False)

def reward_nco(sample_solution, use_cuda=False):
    """
    Args:
        List of length sourceL of [batch_size] Tensors
    Returns:
        Tensor of shape [batch_size] containins rewards
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = torch.zeros([batch_size])
    if use_cuda:
        tour_len = tour_len.cuda()
    for i in range(n-1):
        tour_len += torch.norm(sample_solution[i].data - sample_solution[i+1].data, p=2, dim=1)
    tour_len += torch.norm(sample_solution[n-1].data - sample_solution[0].data, p=2, dim=1)
    return Variable(tour_len, requires_grad=False)


#######################################
# Functions for downloading dataset
#######################################
TSP = namedtuple('TSP', ['x', 'y', 'name'])

GOOGLE_DRIVE_IDS = {
    'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  
    return True

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                 f.write(chunk)

def download_google_drive_file(data_dir, task, min_length, max_length):
    paths = {}
    for mode in ['train', 'test']:
        candidates = []
        candidates.append(
            '{}{}_{}'.format(task, max_length, mode))
        candidates.append(
            '{}{}-{}_{}'.format(task, min_length, max_length, mode))

        for key in candidates:
            print(key)
            for search_key in GOOGLE_DRIVE_IDS.keys():
                if search_key.startswith(key):
                    path = os.path.join(data_dir, search_key)
                    print("Download dataset of the paper to {}".format(path))

                    if not os.path.exists(path):
                        download_file_from_google_drive(GOOGLE_DRIVE_IDS[search_key], path)
                    if path.endswith('zip'):
                        with zipfile.ZipFile(path, 'r') as z:
                            z.extractall(data_dir)
                    paths[mode] = path

    return paths

def read_paper_dataset(paths, max_length):
    x, y = [], []
    for path in paths:
        print("Read dataset {} which is used in the paper..".format(path))
        length = max(re.findall('\d+', path))
        with open(path) as f:
            for l in tqdm(f):
                inputs, outputs = l.split(' output ')
                x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]))
                y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one

    return x, y

def maybe_generate_and_save(self, except_list=[]):
    data = {}

    for name, num in self.data_num.items():
        if name in except_list:
            print("Skip creating {} because of given except_list {}".format(name, except_list))
            continue
        path = self.get_path(name)

        print("Skip creating {} for [{}]".format(path, self.task))
        tmp = np.load(path)
        self.data[name] = TSP(x=tmp['x'], y=tmp['y'], name=name)

def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))

def read_zip_and_update_data(self, path, name):
    if path.endswith('zip'):
        filenames = zipfile.ZipFile(path).namelist()
        paths = [os.path.join(self.data_dir, filename) for filename in filenames]
    else:
        paths = [path]

    x_list, y_list = read_paper_dataset(paths, self.max_length)

    x = np.zeros([len(x_list), self.max_length, 2], dtype=np.float32)
    y = np.zeros([len(y_list), self.max_length], dtype=np.int32)

    for idx, (nodes, res) in enumerate(tqdm(zip(x_list, y_list))):
        x[idx,:len(nodes)] = nodes
        y[idx,:len(res)] = res

    if self.data is None:
        self.data = {}

    print("Update [{}] data with {} used in the paper".format(name, path))
    self.data[name] = TSP(x=x, y=y, name=name)


def create_dataset(problem_size, data_dir):

    def find_or_return_empty(data_dir, problem_size):
        #train_fname1 = os.path.join(data_dir, 'tsp{}.txt'.format(problem_size))
        val_fname1 = os.path.join(data_dir, 'tsp{}_test.txt'.format(problem_size))
        #train_fname2 = os.path.join(data_dir, 'tsp-{}.txt'.format(problem_size))
        val_fname2 = os.path.join(data_dir, 'tsp-{}_test.txt'.format(problem_size))
        
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        else:
            pass
    #         if os.path.exists(train_fname1) and os.path.exists(val_fname1):
    #             return train_fname1, val_fname1
    #         if os.path.exists(train_fname2) and os.path.exists(val_fname2):
    #             return train_fname2, val_fname2
    #     return None, None

    # train, val = find_or_return_empty(data_dir, problem_size)
    # if train is None and val is None:
    #     download_google_drive_file(data_dir,
    #         'tsp', '', problem_size) 
    #     train, val = find_or_return_empty(data_dir, problem_size)

    # return train, val
            if os.path.exists(val_fname1):
                return val_fname1
            if os.path.exists(val_fname2):
                return val_fname2
        return None

    val = find_or_return_empty(data_dir, problem_size)
    if val is None:
        download_google_drive_file(data_dir, 'tsp', '', problem_size)
        val = find_or_return_empty(data_dir, problem_size)

    return val

def create_dataset(
        train_size,
        val_size,
        data_dir,
        tour_len,
        epoch,
        reset=False,
        random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)

    train_task = 'tsp-size-{}-N-{}-train.txt'.format(train_size, tour_len)
    val_task = 'tsp-size-{}-N-{}-val.txt'.format(val_size, tour_len)

    train_fname = os.path.join(data_dir, train_task)
    val_fname = os.path.join(data_dir, val_task)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    else:
        if os.path.exists(train_fname) and os.path.exists(val_fname):
            return train_fname, val_fname

    train_set = open(os.path.join(data_dir, train_task), 'w')
    if not reset:
        val_set = open(os.path.join(data_dir, val_task), 'w')

    def to_string(tensor):
        """
        Convert a a torch.LongTensor 
        of size data_len to a string 
        of integers separated by whitespace
        and ending in a newline character
        """
        mat = ''
        for ii in range(2):
            for jj in range(tour_len - 1):
                mat += '{} '.format(tensor[ii, jj])
            mat += str(tensor[ii,-1]) + '\n'
        return mat

    print('Creating training data set for {}...'.format(train_task))

    # Generate a training set of size train_size
    for i in trange(train_size):
        x = torch.FloatTensor(2, tour_len).uniform_(0, 1)
        train_set.write(to_string(x))
    
    if not reset:
        print('Creating validation data set for {}...'.format(val_task))

        for i in trange(val_size):
            x = torch.FloatTensor(2, tour_len).uniform_(0, 1)
            val_set.write(to_string(x))
        val_set.close()
    train_set.close()

    return train_fname, val_fname

# Dataset
#######################################
class TSPDataset(Dataset):
    
    def __init__(self, dataset_fname=None, use_downloaded_data=False):
        super(TSPDataset, self).__init__()
        
        print(' [*] loading dataset into memory')

        self.data_set = []
        if use_downloaded_data:
            with open(dataset_fname, 'r') as dset:
                for l in tqdm(dset):
                    inputs, outputs = l.split(' output ')
                    sample = torch.zeros(1, )
                    x = np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]).T
                    self.data_set.append(x)
        else:
            with open(dataset_fname, 'r') as dset:
                lines = dset.readlines()
                N = len(lines[0].split())
                ctr = -1
                sample = torch.zeros(N, 2)
                for next_line in tqdm(lines):
                    if ctr < 1:
                        ctr += 1
                    else:
                        ctr = 0
                        self.data_set.append(sample)
                        sample = torch.zeros(N, 2)

                    toks = next_line.split()
                    for idx, tok in enumerate(toks):
                        sample[idx, ctr] = float(tok)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
    
if __name__ == '__main__':
    #paths = download_google_drive_file('data/tsp', 'tsp', '', '50')
    data_dir='/home/pemami/Workspace/deep-assign/data/tsp/icml2018/'
    create_dataset(500000, 1000, data_dir, 15, 0, False, 10)
    create_dataset(500000, 1000, data_dir, 20, 0, False, 10)
    create_dataset(500000, 1000, data_dir, 25, 0, False, 10)
