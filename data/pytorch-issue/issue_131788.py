import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import GPUtil

def select_least_used_gpus(num_gpus):
    gpus = GPUtil.getAvailable(order='memory', limit=num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    print(f'Using GPUs: {gpus}')
    return gpus

# Main loop for training and validation
num_gpus = 4
gpus = select_least_used_gpus(num_gpus)

# Ensure CUDA is available and devices are properly initialized
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
else:
    for gpu in range(len(gpus)):
        try:
            _ = torch.cuda.get_device_properties(gpu)
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing GPU {gpu}: {e}")

# Simulate a training script
class DummyModel(pl.LightningModule):
    def forward(self, x):
        return x

model = DummyModel()
trainer = pl.Trainer(
    max_epochs=1,
    devices=num_gpus,
    accelerator='gpu',
    strategy='ddp',
    logger=pl_loggers.TensorBoardLogger('lightning_logs/'),
    callbacks=[ModelCheckpoint(monitor='val_acc', dirpath='BestModels', filename='best_model', save_top_k=1, mode='max')]
)

trainer.fit(model)