import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

dataset = Planetoid(root='./', name='cora', transform=T.NormalizeFeatures())
g = dataset[0]

print(dataset.num_classes)
print(g)
print(g.keys)
print(g.num_features)

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, g):
        x, edge_index = g.x, g.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = GAT(g.num_features, dataset.num_classes).to(device)
g = g.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def train(graph):
    model.train()

    out = model(graph)
    loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = out.argmax(1)
    train_correct = (pred[graph.train_mask] == graph.y[graph.train_mask]).sum().item()
    train_acc = train_correct / (graph.train_mask.sum().item())

    if i % 10 == 0:
        print("Epoch {:05d}  /  Loss {:.4f}  /  Train_acc {:.4f}".
              format(i, loss.item(), train_acc))

def test(graph):
    model.eval()

    out = model(graph)
    pred = out.argmax(1)
    correct = (pred[graph.test_mask] == graph.y[graph.test_mask]).sum().item()
    test_acc = correct / (graph.test_mask.sum().item())
    print("Test Accuracy :", test_acc)

epoch = 200
for i in range(epoch):
    train(g)

test(g)