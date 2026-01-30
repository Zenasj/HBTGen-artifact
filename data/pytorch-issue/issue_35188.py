tb.add_scalar('loss', loss, global_step=self.train_step)

t0 = time()
s = Summary(value=[Summary.Value(tag='loss', simple_value=float(loss))])
print(time()-t0)

import torch
import time
from torch.utils.tensorboard import SummaryWriter


tb_writer = SummaryWriter(log_dir=f"file://tensorboard",
                          max_queue=100)
for i in range(100):
    tb_writer.add_scalar("train/epoch", i, i)

time.sleep(10)