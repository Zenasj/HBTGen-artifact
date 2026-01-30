import torch
import torch.nn as nn

def config_model(self):
    flags = self.FLAGS.MODEL
    model = self.get_model(flags)
    model.cuda(device=self.device)
    if self.world_size > 1:
      if flags.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
      model = torch.nn.parallel.DistributedDataParallel(
          module=model, device_ids=[self.device],
          output_device=self.device, broadcast_buffers=False,
          find_unused_parameters=flags.find_unused_parameters, static_graph=False)
    if self.is_master:
      print(model)
      if flags.use_compile:
        print('compile the model...')
        model = torch.compile(model)
        print('compiled successfully')
    self.model = model

model=torch.compile(model)

model = torch.compile(model)

...