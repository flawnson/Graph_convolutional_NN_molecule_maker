import torch
import torch.nn.functional as F

import outcome

from scipy.sparse import random
from torch_geometric.nn import GCNConv

examples = outcome.yielder
num_attr = outcome.get_network_params(examples)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_attr, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.sigmoid(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01, weight_decay=5e-4)

discriminator.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = discriminator(data)
    loss = F.BCELoss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

discriminator.eval()
_, pred = discriminator(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))