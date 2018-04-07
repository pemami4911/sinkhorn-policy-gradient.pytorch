from math import log10, floor
import torch
from sklearn.utils.linear_assignment_ import linear_assignment

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

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def parallel_matching(batch):
    perms = []
    (m, n, n) = batch.shape
    for i in range(m):
        perm = torch.zeros(n, n)
        matching = linear_assignment(-batch[i])
        perm[matching[:,0], matching[:,1]] = 1
        perms.append(perm)
    return perms


def memory_usage():
    return ((int(open('/proc/self/statm').read().split()[1]) * 4096.) / 1000000.)

if __name__ == '__main__': 
    torch.random.manual_seed(1)
    ones = torch.ones(3,1)
    x = torch.zeros(3,3).uniform_()
    x = torch.exp(x / 0.1)
    #x = torch.ones(3,3)
    print(x)

    log_scale_res_2 = torch.log(x) - logsumexp(x, dim=0, keepdim=True)
    print("Log scale res 2 {}".format(log_scale_res_2))
    print("exp(log_scale_res_2): {}".format(torch.exp(log_scale_res_2)))

    rn = torch.div(x , torch.matmul(torch.matmul(x,ones), torch.t(ones)))
    print("Unstable: {}".format(rn))
