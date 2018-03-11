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
