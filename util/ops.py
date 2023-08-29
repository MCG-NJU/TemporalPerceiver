
import torch

def prop_relative_to_absolute(x, base, window_size, interval):
    num_samples = x.shape[1]
    base = base.unsqueeze(1).repeat(1, num_samples).cuda()
    x = x.view(-1,num_samples)
    b = [x * window_size * interval + base]
    return torch.stack(b, dim=-1)

