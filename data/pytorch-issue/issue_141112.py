import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=256, embed_dim=1024, upsample_rate=4, explode=False):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.explode = explode
        self.fc_upsample = nn.Linear(in_dim, upsample_rate * embed_dim)
        self.fc_outer = nn.Linear(embed_dim, in_dim)
    
    def forward(self, in_tensor):
        ### Upsample
        tensor = self.fc_upsample(in_tensor)

        ### ***BUG*** Reshaping nested tensor directly results in exploding cached memory usage (Why???)
        if self.explode:
            tensor = tensor.view(tensor.size(0), tensor.size(1), self.upsample_rate, self.embed_dim)
            tensor = self.fc_outer(tensor).flatten(-2)
            return tensor
        
        ### ***WORK AROUND*** This indirect reshaping results in fixed cached memory usage
        else:
            tensor = tensor.values()
            tensor = tensor.view(tensor.size(0), self.upsample_rate, self.embed_dim)
            tensor = self.fc_outer(tensor).flatten(-2)
            return torch.nested.nested_tensor_from_jagged(tensor, offsets=in_tensor.offsets())

### Setup
in_dim = 512
embed_dim = 512
max_seq_len = 100
device = 'cuda:0'

### Change explode to False to observe stable behavior
net = MLP(in_dim=in_dim, embed_dim=embed_dim, upsample_rate=4, explode=True).to(device)
# net = MLP(in_dim=in_dim, embed_dim=embed_dim, upsample_rate=4, explode=False).to(device)

optim = torch.optim.Adam(net.parameters())

### Train
num_steps = 1000
batch_size = 64
for step in range(num_steps):
    ### Forward, backward, optimize
    optim.zero_grad()
    in_tensor = [torch.rand(size=[seq_len, in_dim]) for seq_len in torch.randint(max_seq_len, size=(batch_size,))]
    in_tensor = torch.nested.as_nested_tensor(in_tensor, layout=torch.jagged).to(device)
    loss = net(in_tensor).values().pow(2).mean() ### Dummy weight decay loss
    loss.backward()
    optim.step()

    ### Record average parameter magnitude
    with torch.no_grad():
        param_mag = 0.
        count = 0
        for param in net.parameters():
            param_mag = param_mag + param.abs().sum().item()
            count = count + param.numel()
        param_mag /= count

    ### Print memory usage
    max_allocated = torch.cuda.max_memory_allocated(0) // 1024 ** 3
    max_reserved = torch.cuda.max_memory_reserved(0) // 1024 ** 3
    print(f'>>> Step={step}, VRAM={max_allocated:.3f}<{max_reserved:.3f}GiB, param_mag={param_mag:.3f}')