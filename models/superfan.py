import torch
import torch.nn as nn
import torch.nn.functional as F


def glorot_tensor(shape):
    ''' Builds torch tensor randomly initialized with glorot '''
    return nn.Parameter(nn.init.xavier_uniform_(torch.zeros(shape)),
                        requires_grad=True)


class Fan_Arm(nn.Module):
    '''
    Arm of Superfan network. Data is passed into arm nodes for representation
    and then pooled by pool node. Returns scalar of sub-classification.
    '''
    def __init__(self, in_dim, arm_size):
        super(Fan_Arm, self).__init__()
        self.arm = glorot_tensor(shape=(arm_size, in_dim))
        self.pool = torch.zeros(arm_size)

    def forward(self, x):
        return torch.dot(self.pool, torch.mv(self.arm, x))


class Superfan(nn.Module):
    '''
    Applies "superfan" architecture to regression.
    '''
    def __init__(self, in_dim, out_dim, arm_num, arm_size):
        super(Superfan, self).__init__()
        self.arm = glorot_tensor(shape=(in_dim, arm_size))
        self.pool = glorot_tensor(shape=(arm_size,))
        self.center = glorot_tensor(shape=(arm_num, out_dim))
        self.arms = nn.ModuleList([])

    def criterion(self, fx, y, cov_penalty):
        '''
        Criterion is sum of binary crossentropy between fx and y with scaled
        sum of covariances between each arm latent subspace.
        '''
        return torch.log(torch.dot(fx, y))
