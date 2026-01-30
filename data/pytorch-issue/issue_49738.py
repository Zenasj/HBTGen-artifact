import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = nn.Conv2d(in_size, out_size, 1)
        self.norm = nn.LayerNorm(64)

    def forward(self, inp):
        # checkpoint cross attn
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        output = checkpoint(create_custom_forward(self.conv), inp)
        output = output.permute(0, 2, 3, 1)  # permute so we have channel in the end
        output = checkpoint(create_custom_forward(self.norm), output)

        return output


in_size = 32
out_size = 64
net = Net(in_size, out_size).cuda()
opt = torch.optim.SGD(net.parameters(), lr=0.001)

net.train()

# without amp
for epoch in range(10):
    inp = torch.randn([1, 32, 32, in_size], requires_grad=True).cuda()
    output = net(inp)
    loss = output.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()  # set_to_none=True here can modestly improve performance

print("finish training without amp")

# with amp
scaler = torch.cuda.amp.GradScaler()

for epoch in range(10):
    inp = torch.randn([1, 32, 32, in_size], requires_grad=True).cuda()
    with torch.cuda.amp.autocast():
        output = net(inp)
        loss = output.sum()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()  # set_to_none=True here can modestly improve performance

print("finish training with amp")