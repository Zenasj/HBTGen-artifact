import torch

g = torch.cuda._Graph()
g.capture_begin()
g.capture_end()
g.replay()

# user must run a few warmup iterations first, e.g. to get cudnn benchmarking out of the way
g = torch.cuda._Graph()
with torch.cuda.graph(g):
    static_loss = net(static_in)
    static_loss.backward()
    optimizer.step()

for d in data:
    static_in.copy(d)
    g.replay() # plays the whole fwd, bwd, step. Efficient!
    # you don't even need zero_grad because replay() refills the static .grad attributes in place.