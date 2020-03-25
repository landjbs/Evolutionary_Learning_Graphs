import math
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

    # visualization
    def visualize(self, c_dict=None):
        '''
        Visualizes network with variable number arms and arm sizes.
        Args:
            c_dict:      Dict mapping 'center', 'pool_i', and 'arm_i' to colors
        '''
        if not c_dict:
            c_dict = {'center' : 'black'}
            for i in range(self.arm_num):
                c_dict[f'pool_{i}'] = 'blue'
                c_dict[f'arm_{i}'] = ['red' for _ in range(self.arm_size)]
        # basic params
        c = (0, 0)                              # center coordinates
        c_rad = 0.2                             # little radius around center
        r = 1                                   # max length of arm
        plt.ylim((-r, r))
        plt.xlim((-r, r))
        plt.scatter(c[0], c[1], c=c_dict['center'], zorder=20)
        # choose sizing for fan size
        theta = (2 * math.pi / self.arm_num)    # angle between each arm
        theta_delt = (theta / 3)                # angle between arm and pool
        r_delt = ((r - c_rad) / self.arm_size)  # distance between arm nodes
        pool_r = (r / 2)                        # distance center -> pool
        # plot each arm
        theta_acc = 0                           # accumulator for angles
        for arm in range(self.arm_num):
            pool_x = pool_r * math.cos(theta_acc)
            pool_y = pool_r * math.sin(theta_acc)
            plt.scatter(pool_x, pool_y, color=c_dict[f'pool_{arm}'], zorder=10)
            plt.plot([pool_x, c[0]], [pool_y, c[1]])
            theta_acc += theta_delt
            r_acc = c_rad
            for arm_node in range(self.arm_size):
                arm_x = r_acc * math.cos(theta_acc)
                arm_y = r_acc * math.sin(theta_acc)
                plt.scatter(arm_x, arm_y, color=c_dict[f'arm_{arm}'][arm_node],
                            zorder=10)
                # print([arm_x, arm_y], [0, 0])
                plt.plot([arm_x, pool_x], [arm_y, pool_y], zorder=5)
                r_acc += r_delt
            theta_acc += (theta - theta_delt)
        plt.title('Network')
        plt.axis('off')
        plt.show()

    def color_node(self, fx):
        ''' Colors node as function of outbound activation '''
        return 

    def visualize_signal_prop(self, x):
        '''
        Visualizes propagation of signal through network
        '''
        c_dict = {}
        pools = []
        for i_arm, arm in enumerate(self.arms):
            pooling, encodings = arm(x)
            c_dict[f'pool_{i_arm}'] = pooling.detach().numpy()
            c_dict[f'arm_{i_arm}'] = [x.detach().numpy() for x in encodings]
            pools.append(pooling)
        fx, _ = self(x)
        c_dict[f'center'] = fx.detach().numpy()
        print(c_dict)
        self.visualize(c_dict)
