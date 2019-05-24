# TODO: The damn thing still keeps showing the same error, either the features.py data making algorithm is flawed or this model is doing something funky
#       A potential source of the problem is in the dataset, datasetpoint, and data variables in the model training section of this program 

import os
import outcome
import features
import torch

import numpy as np
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.sparse import random
from scipy import stats
from numpy.random import normal

"""DATA IMPORTING"""
num_attr, list_num_atoms = outcome.get_network_params()
print(num_attr)
print(features.get_num_classes(features.get_atom_symbols(features.suppl)))

train, val, test = outcome.splitter(list(outcome.yielder()))
edge_index = train[0].edge_index
x = train[0].x

"""MODEL ARCHITECTURE"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 5)

    def forward(self, data):
        num_attr, list_num_atoms = outcome.get_network_params()
        train, val, test = outcome.splitter(list(outcome.yielder()))
        edge_index = train[0].edge_index
        x = train[0].x

        x = self.conv1(x.float(), edge_index.long())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x.float(), edge_index.long())

        return F.log_softmax(x, dim=1)

"""MODEL TRAINING"""
# torch.set_default_tensor_type('torch.DoubleTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device=device, dtype=torch.float)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
# for epoch in range(200):
for datapoint in train:
    data = datapoint.to(device)
    optimizer.zero_grad()
    out = model(datapoint)
    loss = F.nll_loss(out[data.train], datapoint.y[data.train])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))