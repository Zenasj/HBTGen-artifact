import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

SHAPE = [16, 3, 256, 256]
TRAIN_STEPS = 200
VALID_STEPS = 2000

def conv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, 3, padding=1)

def relu(x):
  return F.relu(x)

def pool(x):
  return F.max_pool2d(x, 2, 2)

def upsample(x):
  return F.interpolate(x, scale_factor=2, mode='nearest')

def concat(a, b):
  return torch.cat((a, b), 1)

class UNet(nn.Module):
  def __init__(self, ic=3, oc=3):
    super(UNet, self).__init__()

    ec1, ec2, ec3, ec4, ec5, dc4, dc3, dc2, dc1a, dc1b = 32, 48, 64, 80, 96, 112, 96, 64, 64, 32

    self.enc_conv0  = conv(ic,      ec1)
    self.enc_conv1  = conv(ec1,     ec1)
    self.enc_conv2  = conv(ec1,     ec2)
    self.enc_conv3  = conv(ec2,     ec3)
    self.enc_conv4  = conv(ec3,     ec4)
    self.enc_conv5a = conv(ec4,     ec5)
    self.enc_conv5b = conv(ec5,     ec5)
    self.dec_conv4a = conv(ec5+ec3, dc4)
    self.dec_conv4b = conv(dc4,     dc4)
    self.dec_conv3a = conv(dc4+ec2, dc3)
    self.dec_conv3b = conv(dc3,     dc3)
    self.dec_conv2a = conv(dc3+ec1, dc2)
    self.dec_conv2b = conv(dc2,     dc2)
    self.dec_conv1a = conv(dc2+ic,  dc1a)
    self.dec_conv1b = conv(dc1a,    dc1b)
    self.dec_conv0  = conv(dc1b,    oc)

  def forward(self, input):
    x = relu(self.enc_conv0(input))
    x = relu(self.enc_conv1(x))
    x = pool1 = pool(x)
    x = relu(self.enc_conv2(x))
    x = pool2 = pool(x)
    x = relu(self.enc_conv3(x))
    x = pool3 = pool(x)
    x = relu(self.enc_conv4(x))
    x = pool(x)
    x = relu(self.enc_conv5a(x))
    x = relu(self.enc_conv5b(x))
    x = upsample(x)
    x = concat(x, pool3)
    x = relu(self.dec_conv4a(x))
    x = relu(self.dec_conv4b(x))
    x = upsample(x)
    x = concat(x, pool2)
    x = relu(self.dec_conv3a(x))
    x = relu(self.dec_conv3b(x))
    x = upsample(x)
    x = concat(x, pool1)
    x = relu(self.dec_conv2a(x))
    x = relu(self.dec_conv2b(x))
    x = upsample(x)
    x = concat(x, input)
    x = relu(self.dec_conv1a(x))
    x = relu(self.dec_conv1b(x))
    x = self.dec_conv0(x)
    return x

def demo_basic():
  dist.init_process_group("nccl")
  rank = dist.get_rank()
  print(f"Start on rank {rank}.")

  device_id = rank % torch.cuda.device_count()
  model = UNet().to(device_id)
  model = torch.compile(model, mode='reduce-overhead')
  ddp_model = DDP(model, device_ids=[device_id])

  loss_fn = nn.MSELoss()
  loss_fn = torch.compile(loss_fn, mode='reduce-overhead')
  optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

  if rank == 0:
    print('Train: ', end='')

  for i in range(TRAIN_STEPS):
    torch.compiler.cudagraph_mark_step_begin()

    input  = torch.randn(SHAPE, dtype=torch.float32, device=device_id)
    target = torch.randn(SHAPE, dtype=torch.float32, device=device_id)

    optimizer.zero_grad()
    output = ddp_model(input)
    loss_fn(output, target).backward()
    optimizer.step()

    if rank == 0 and i % 10 == 0:
      print('.', end='')

  if rank == 0:
    print('\nValid: ', end='')

  model.eval()

  for i in range(VALID_STEPS):
    with torch.no_grad():
      torch.compiler.cudagraph_mark_step_begin()

      input  = torch.randn(SHAPE, dtype=torch.float32, device=device_id)
      target = torch.randn(SHAPE, dtype=torch.float32, device=device_id)

      output = ddp_model(input)
      loss_fn(output, target)

      if rank == 0 and i % 10 == 0:
        print('.', end='')

  if rank == 0:
    print('\nDone')

  dist.destroy_process_group()

if __name__ == "__main__":
  demo_basic()