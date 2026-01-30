import torch
import torch.nn as nn

# Same definitions for Module classes and constants as in the tutorial code.
...

dropout = 0.0

num_gpus = 2
partition_len = ((nlayers - 1) // num_gpus) + 1



# build layers
encoder = Encoder(ntokens, emsize, dropout)
transformer_layers = [TransformerEncoderLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers)]
decoder = Decoder(ntokens, emsize)

encoder1 = Encoder(ntokens, emsize, dropout)
transformer_layers1 = [TransformerEncoderLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers)]
decoder1 = Decoder(ntokens, emsize)

# use the same set of parameters
for src_net, dest_net in zip([encoder, decoder] + transformer_layers, [encoder1, decoder1] + transformer_layers1):
    params_dict = src_net.state_dict()
    params_dict_copy = {key:params_dict[key].clone() for key in params_dict.keys()}
    dest_net.load_state_dict(params_dict_copy)

# build pipeline
net = nn.Sequential(
    nn.Sequential(*([encoder] + transformer_layers[:partition_len])).cuda(0),
    nn.Sequential(*(transformer_layers[partition_len:] + [decoder])).cuda(num_gpus - 1)
)
# Or without pipeline parallelism:
# net = nn.Sequential(*([encoder] + transformer_layers + [decoder])).cuda(0)
net1 = nn.Sequential(
    nn.Sequential(*([encoder1] + transformer_layers1[:partition_len])).cuda(0),
    nn.Sequential(*(transformer_layers1[partition_len:] + [decoder1])).cuda(num_gpus - 1)
)
net = Pipe(net, chunks=2)
net1 = Pipe(net1, chunks=8)

# test using a random vector
random_input = torch.randint(0, 10000, [bptt, batch_size]).cuda(0)
net_out = net(random_input).local_value()
net1_out = net1(random_input).local_value()

print(torch.max(torch.abs(net_out - net1_out)))