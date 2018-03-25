from math import log10, floor
import torch

def str2bool(v):
      return v.lower() in ('true', '1')

def cudify(x, use_cuda):
    """
    Args 
        x: input Tensor
        use_cuda: boolean   
    """
    if use_cuda:
        return x.cuda()
    else:
        return x

def loss_dt_check(losses):
    """
    Compute an estimate of the rate 
    of change of the loss given a 
    deque of loss values 
    """
    dt = []
    for i in range(1, len(losses)):
        dt.append(losses[i] - losses[i-1])
    return sum(dt) / (len(losses) - 1)

def copy_model_params(source, target):
    target_p = list(target.parameters())
    source_p = list(source.parameters())
    n = len(source_p)
    for i in range(n):
        target_p[i].data[:] = source_p[i].data[:]

def round_to_2(x):
    return round(x, -int(floor(log10(abs(x))))+1)

def byte_tensor_to_index(x):
    """
    Convert a torch.ByteTensor of size [batch_size, 1]
    to a torch.LongTensor containing only nonzero elements
    of x converted to the corresponding index values.
    """
    idx_tensor = []
    for i in range(x.size()[0]):
        if x.data[i][0] == 1:
            idx_tensor.append(i)
    return torch.LongTensor(idx_tensor)

