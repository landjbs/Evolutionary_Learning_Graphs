import torch

from models.superfan import Superfan

x = torch.ones(10)

z = Superfan(in_dim=10, out_dim=2, arm_num=4, arm_size=5, lr=0.001)

print(z(x))
