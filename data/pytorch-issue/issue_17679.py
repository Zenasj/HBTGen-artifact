import torch.nn as nn

import torch

emb = torch.nn.Embedding(10,10)
emb2 = torch.nn.Embedding(10,10)

optim = torch.optim.Adagrad(emb.parameters())
print(optim.state[emb.weight])  # already initialized

optim.add_param_group({'params': emb2.parameters()})
print(optim.state[emb2.weight])  # empty dict

loss = emb2.weight.sum() + emb.weight.sum()
loss.backward()
optim.step()  # raised KeyError

model = make_model(config)
optimizer = Adagrad(model.parameters(), lr=config.lr)
optimizer.share_memory()
edges = EdgeReader(config.edge_paths[0]).read()
pool = torch.multiprocessing.Pool(config.num_workers)
def train(model, edges, optimizer):
    model.zero_grad()
    loss = model(edges)
    loss.backward()
    optimizer.step()
pool.starmap(train, [(model, edges[i:i + 1000], optimizer) for i in range(0, len(edges), 1000)])