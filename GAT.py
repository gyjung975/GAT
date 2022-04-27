import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import dgl
from dgl import DGLGraph
from dgl.data import CitationGraphDataset

citeseer = CitationGraphDataset('citeseer')
cora = CitationGraphDataset('cora')

import networkx as nx
# nx_G = cora.graph.to_undirected()
# pos = nx.kamada_kawai_layout(nx_G)
# display(nx.draw(nx_G, pos, with_labels=False, node_size = 0.01, node_color='#00b4d9'))

def load_citeseer_data():
    data = citeseer
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = DGLGraph(data.graph)
    
    # g = data.graph
    # g.remove_edges_from(nx.selfloop_edges(g))
    # g = DGLGraph(g)
    # g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask

def load_cora_data():
    data = cora
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = DGLGraph(data.graph)

    # g = data.graph
    # g.remove_edges_from(nx.selfloop_edges(g))
    # g = DGLGraph(g)
    # g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

data = citeseer
g, features, labels, mask = load_citeseer_data()
# data = cora
# g, features, labels, mask = load_cora_data()

model = GAT(g,
            in_dim=features.size()[1],
            hidden_dim=8,
            # citeseer
            out_dim=6,
            # cora
            # out_dim=8,
            num_heads=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)

dur = []

for epoch in range(1000):
    if epoch >= 3:
        t0 = time.time()

    logits = model(features)
    logp = F.log_softmax(logits, 1)

    loss = F.nll_loss(logp[mask], labels[mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    if epoch % 100 == 0:
        print("Epoch {:05d}  /  Loss {:.4f}  /  Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))

mask = torch.BoolTensor(data.test_mask)

pred = np.argmax(logp[mask].detach().numpy(), axis = 1)
answ = labels[mask].numpy()
print(np.sum([1 if pred[i] == answ[i] else 0 for i in range(len(pred))]) / len(pred) * 100)