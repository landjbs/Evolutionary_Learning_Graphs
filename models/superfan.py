import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange


def glorot_tensor(shape):
    ''' Builds torch tensor randomly initialized with glorot '''
    return nn.Parameter(nn.init.xavier_uniform_(torch.ones(shape)),
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
        self.arm_bias = torch.ones(arm_size, requires_grad=True)
        # pool weights
        self.pool_weights = torch.ones(arm_size)
        self.pool_bias = torch.ones(1, requires_grad=True)
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
    def __init__(self, in_dim, out_dim, arm_num, arm_size):
        super(Superfan, self).__init__()
        # training params
        self.lr = 0.01
        self.corr_term = 0.001
        self.regularization_term = 0.001
        # cache
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.arm_num = arm_num
        self.arm_size = arm_size
        # list of arms
        self.arms = nn.ModuleList([Fan_Arm(in_dim, arm_size, id)
                                   for id in range(arm_num)])
        # weights for center
        self.weights = glorot_tensor(shape=(out_dim, arm_num))
        self.bias = torch.ones(out_dim, requires_grad=True)
        # activation for center
        self.non_linearity = F.relu
        # optimizers
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        ''' Returns prediction vector from center and encodings from arms '''
        pools, encodings = zip(*(arm(x) for arm in self.arms))
        aggregate_pooling = torch.cat(pools)
        fx = torch.mv(self.weights, aggregate_pooling) + self.bias
        fx = self.non_linearity(fx)
        return fx, encodings

    def correlations(self):
        ''' Calculates cumulative correlations of arm weights '''
        corr = 0
        vars = [torch.var(arm.arm_weights) for arm in self.arms]
        for i, i_arm in enumerate(self.arms):
            for j, j_arm in enumerate(self.arms):
                if (i != j):
                    i_arm, j_arm = i_arm.arm_weights, j_arm.arm_weights
                    i_mean, j_mean = torch.mean(i_arm), torch.mean(j_arm)
                    cov = torch.dot((i_arm - i_mean), (j_arm - j_mean))
                    corr += (cov / torch.sqrt(vars[i], vars[j]))
        norm_corr =  (corr / self.arm_num)
        return norm_corr

    def l_p_norm(self, p):
        ''' Calcs p norm of all network weights '''
        norm = 0
        for mat in self.parameters(recurse=True):
            norm += torch.sum(mat.pow(p))
        return norm

    def criterion(self, fx, y, encodings):
        '''
        Criterion is sum of binary crossentropy between fx and y with scaled
        sum of correlations across latent subspaces.
        '''
        if self.out_dim==1:
            delta_penalty = torch.pow(fx-y, 2)
        else:
            delta_penalty = -torch.log(fx, y)
        # corr_penalty = self.corr_term * self.correlations()
        regularization_penalty = self.regularization_term * self.l_p_norm(2)
        return delta_penalty + regularization_penalty

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

    def train(self, data, epochs, batch_size):
        data_size = len(data)
        losses = []
        for _ in trange(epochs):
            batch = [data[i] for i in torch.randint(0, data_size,
                                                    size=(batch_size,))]
            loss = self.train_on_batch(batch)
            losses.append(loss.item())
        plt.plot(losses)
        plt.title('Losses')
        plt.show()

    def visualize(self):
        plt.ylim((0,self.size))
        plt.xlim((0,self.size))
