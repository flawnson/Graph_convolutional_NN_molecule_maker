import torch
import os

import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

directory = r"C:\Users\Flawnson\Documents\Project Seraph & Cherub\Project Outcome\datasets\processed"

for subdir, dirs, files in os.walk(directory):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".pt"):
                print(torch.load(str(filepath)))

                        # =================================================================================== #
                        #                                NETWORK ARCHITECTURE                                 #
                        # =================================================================================== #

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()

# model.eval()
# _, pred = model(data).max(dim=1)
# correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
# acc = correct / data.test_mask.sum().item()
# print('Accuracy: {:.4f}'.format(acc))
# >>> Accuracy: 0.8150