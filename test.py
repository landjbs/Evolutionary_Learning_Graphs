import torch

from models.superfan import Superfan


def make_batch(in_dim, out_dim, n):
    assert out_dim==1, 'multidim outs not yet supported'
    batch = []
    for _ in range(n):
        x = torch.rand(in_dim)
        y = torch.sum(x)
        batch.append((x, y))
    return batch


in_dim = 2
out_dim = 1

batch = make_batch(in_dim, out_dim)

z = Superfan(in_dim=10, out_dim=2, arm_num=4, arm_size=5, lr=0.001)

print(z(x))
