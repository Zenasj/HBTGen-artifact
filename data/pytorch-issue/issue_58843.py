import torch
import torch.nn as nn

Channel = 2
torch.manual_seed(7)
instance_model = nn.InstanceNorm2d(Channel, affine=True, track_running_stats=True)

torch.manual_seed(7)
batch_model = nn.BatchNorm2d(Channel, affine=True, track_running_stats=True)

# Do some "training".
instance_optimizer = torch.optim.Adam(instance_model.parameters())
batch_optimizer = torch.optim.Adam(batch_model.parameters())
for _ in range(10):
    dummy_input = torch.randn((3,Channel,1,3))
    instance_optimizer.zero_grad()
    batch_optimizer.zero_grad()
    instance_model(dummy_input).sum().backward()
    batch_model(dummy_input).sum().backward()
    instance_optimizer.step()
    batch_optimizer.step()

instance_model.eval()
batch_model.eval()

with torch.no_grad():
    instance_out_1 = instance_model(dummy_input)
    instance_out_2 = instance_model(dummy_input)

with torch.no_grad():
    batch_out_1 = batch_model(dummy_input)
    batch_out_2 = batch_model(dummy_input)

print("result_1:", (instance_out_1 == batch_out_1).to(torch.bool))
print("result_2:", (instance_out_2 == batch_out_2).to(torch.bool))