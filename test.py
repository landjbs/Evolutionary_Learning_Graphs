import torch

from models.superfan import Superfan


def make_batch(n, in_dim, out_dim):
    assert out_dim==1, 'multidim outs not yet supported'
    batch = []
    for _ in range(n):
        x = torch.rand(in_dim)
        y = torch.tensor([torch.sum(x)])
        batch.append((x, y))
    return batch


in_dim = 1
out_dim = 1

batch = make_batch(10000, in_dim, out_dim)

# import matplotlib.pyplot as plt
# x = [b[0].item() for b in batch]
# y = [b[1].item() for b in batch]
# plt.scatter(x, y)
# plt.show()

z = Superfan(in_dim=in_dim, out_dim=out_dim, arm_num=4, arm_size=5)

z.train(batch, 100, 10)
