import torch

from models.superfan import Fan_Arm

x = torch.ones(10)

z = Fan_Arm(10, 3)

print(z(x))
