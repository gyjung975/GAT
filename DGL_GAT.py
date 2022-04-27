import dgl
from dgl.data import register_data_args
from dgl.data import CitationGraphDataset
# from dgl.data import CoraGraphDataset, CiteseerGraphDataset

dataset = CitationGraphDataset('cora')
# dataset = CoraGraphDataset()
g = dataset[0]

print(dataset.num_classes)
print(g)

n_classes = dataset.num_classes
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
labels = g.ndata['label']
features = g.ndata['feat']

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0],
                                       feat_drop, attn_drop, negative_slope, False, self.activation))

        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l],
                                           feat_drop, attn_drop, negative_slope, residual, self.activation))

        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1],
                                       feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features):
        h = features
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        out = self.gat_layers[-1](self.g, h).mean(1)
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

num_heads = 2
num_out_heads = 1
num_layers = 1
heads = ([num_heads] * num_layers) + [num_out_heads]
print(heads)

model = GAT(g,
            in_dim=features.shape[1],
            num_classes=n_classes,
            heads=heads,
            activation=F.elu,
            num_layers=1,
            num_hidden=8,
            feat_drop=0.6,
            attn_drop=0.6,
            negative_slope=0.2,
            residual=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def evaluate(model, features, mask):
    model.eval()
    with torch.no_grad():
        out = model(features)
        out = out[mask]
        return out

dur = []
for epoch in range(200):
    model.train()

    if epoch >= 3:
        t0 = time.time()

    out = model(features)
    loss = criterion(out[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    pred = out.argmax(1)
    train_correct = (pred[train_mask] == labels[train_mask]).sum().item()
    train_acc = train_correct / len(labels[train_mask])

    val_out = evaluate(model, features, val_mask)
    val_pred = val_out.argmax(1)
    val_correct = (val_pred == labels[val_mask]).sum().item()
    val_acc = val_correct / len(labels[val_mask])

    if epoch % 10 == 0:
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} | ValAcc {:.4f}".
          format(epoch, np.mean(dur), loss.item(), train_acc, val_acc))