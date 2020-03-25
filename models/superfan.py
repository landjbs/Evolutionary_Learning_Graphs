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
    def __init__(self, in_dim, arm_size, id=None):
        super(Fan_Arm, self).__init__()
        # arm weights
        self.arm_weights = glorot_tensor(shape=(arm_size, in_dim))
        self.arm_bias = torch.zeros(arm_size, requires_grad=True)
        # pool weights
        self.pool_weights = torch.zeros(arm_size)
        self.pool_bias = torch.zeros(1, requires_grad=True)
        # vars
        self.non_linearity = F.relu
        self.id = id

    def forward(self, x):
        arm_encoding = torch.mv(self.arm_weights, x) + self.arm_bias
        pooled = torch.dot(self.pool_weights, arm_encoding) + self.pool_bias
        return self.non_linearity(pooled), arm_encoding


class Superfan(nn.Module):
    '''
    Applies "superfan" architecture to regression.
    '''
    def __init__(self, in_dim, out_dim, arm_num, arm_size, lr):
        super(Superfan, self).__init__()
        # list of arms
        self.arms = nn.ModuleList([Fan_Arm(in_dim, arm_size, id)
                                   for id in range(arm_num)])
        # weights for center
        self.weights = glorot_tensor(shape=(out_dim, arm_num))
        self.bias = torch.zeros(out_dim, requires_grad=True)
        # activation for center
        self.non_linearity = F.relu
        # optimizers
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # training params
        self.lr = 0.0001
        self.cov_term = 0.001
        self.l2_term = 0.001

    def forward(self, x):
        ''' Returns prediction vector from center and encodings from arms '''
        pools, encodings = zip(*(arm(x) for arm in self.arms))
        aggregate_pooling = torch.cat(pools)
        fx = torch.mv(self.weights, aggregate_pooling) + self.bias
        fx = self.non_linearity(fx)
        return fx, encodings

    def criterion(self, fx, y, encodings):
        '''
        Criterion is sum of binary crossentropy between fx and y with scaled
        sum of covariances across latent subspaces.
        '''
        cov = torch.sum()
        return torch.log(torch.dot(fx, y))

    def train_on_batch(self, batch):
        '''
        Trains all weights on batch of x, y pairs
        '''
        self.optimizer.zero_grad()
        loss = 0
        for x, y in batch:
            fx, encodings = self(x)
            loss += self.criterion(fx, y, encodings)
        loss.backward()
        self.optimizer.step()
        return loss

    def train()
