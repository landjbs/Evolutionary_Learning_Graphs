import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import Base_Model



def glorot_tensor(shape):
    ''' Builds torch tensor randomly initialized with glorot '''
    return nn.Parameter(nn.init.xavier_uniform_(torch.zeros(shape)),
                        requires_grad=True)


class Superfan(Base_Model):
    '''
    Applies "superfan" architecture to regression.
    '''
    def __init__(self, in_dim, out_dim, arm_num):
        super(Base_Model, self).__init__()
        self.arm = torch.tensor()

    def criterion(self, fx, y, cov_penalty):
        '''
        Criterion is sum of binary crossentropy between fx and y with scaled
        sum of covariances between each arm latent subspace.
        '''
        return torch.log(torch.dot(fx, y)) +
