import torch.nn as nn

class MyModel(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.generator = GeneratorNetwork()
        self.loss = LossWithDiscriminatorNetwork()

def on_train_batch_end(self, outputs, batch, batch_idx):
        opts = self.optimizers()
        scheds = self.lr_schedulers()
        current_cycle = (batch_idx // self.n_batches_per_optimizer) % len(opts)
        opt = opts[current_cycle]

        with opt.toggle_model():
            self.manual_backward(self.computed_loss)
            if (batch_idx + 1) % self.n_batches_per_optimizer == 0:
                opt.step()
                opt.zero_grad()

                if scheds is not None:
                    # None if we're using defaults ... not None if we're using a scheduler.
                    # If we're using a scheduler we have to step it forward manually.
                    scheds[current_cycle].step()

def toggle_optimizer(self, optimizer: Union[Optimizer, LightningOptimizer]) -> None:
        """Makes sure only the gradients of the current optimizer's parameters are calculated in the training step to
        prevent dangling gradients in multiple-optimizer setup.

        It works with :meth:`untoggle_optimizer` to make sure ``param_requires_grad_state`` is properly reset.

        Args:
            optimizer: The optimizer to toggle.

        """
        # Iterate over all optimizer parameters to preserve their `requires_grad` information
        # in case these are pre-defined during `configure_optimizers`
        param_requires_grad_state = {}
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                for param in group["params"]:
                    # If a param already appear in param_requires_grad_state, continue
                    if param in param_requires_grad_state:
                        continue
                    param_requires_grad_state[param] = param.requires_grad
                    param.requires_grad = False

        # Then iterate over the current optimizer's parameters and set its `requires_grad`
        # properties accordingly
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.requires_grad = param_requires_grad_state[param]
        self._param_requires_grad_state = param_requires_grad_state

"""
File: test_fsdp.py
Description: Minimal example of FSDP failure with unused parameters
"""

import os

import torch
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset


class AdversarialModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.generator = GeneratorNetwork()
        self.discriminator = DiscriminatorNetwork()
        self.automatic_optimization = False

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        current_cycle = batch_idx % len(opts)

        if current_cycle == 0:
            #  compute loss from generator
            self.computed_loss = self.generator(batch).mean()
        else:
            # compute loss from discriminator
            self.computed_loss = self.discriminator(batch).mean()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        opts = self.optimizers()
        current_cycle = batch_idx % len(opts)
        opt = opts[current_cycle]

        with opt.toggle_model():
            self.manual_backward(self.computed_loss)
            opt.step()
            opt.zero_grad()

    def validation_step(self, batch, batch_idx):
        generator_loss = self.generator(batch).mean()
        discriminator_loss = self.discriminator(batch).mean()
        self.log("valid_generator_loss", generator_loss)
        self.log("valid_discriminator_loss", discriminator_loss)

    def configure_optimizers(self):
        return [
            torch.optim.SGD(self.generator.parameters(), lr=0.1),
            torch.optim.SGD(self.discriminator.parameters(), lr=0.1)
        ]


class GeneratorNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)


class DiscriminatorNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = AdversarialModel()
    trainer = Trainer(default_root_dir=os.getcwd(),
                      limit_train_batches=10,
                      limit_val_batches=10,
                      num_sanity_val_steps=0,
                      max_epochs=1,
                      enable_model_summary=False,
                      num_nodes=1,
                      devices=8,
                      strategy='fsdp',
                      enable_progress_bar=True)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    run()

import os
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


#----------------------------
# MODEL
#----------------------------
class AdversarialModel(nn.Module):

    def __init__(self):
        super(AdversarialModel, self).__init__()
        self.generator = GeneratorNetwork()
        self.discriminator = DiscriminatorNetwork()

    def forward(self, x, mode='generator'):
        if mode == 'generator':
            return self.generator(x)
        elif mode == 'discriminator':
            return self.discriminator(x)

    def generator_parameters(self, recurse: bool = True):
        return self.generator.parameters(recurse=recurse)

    def discriminator_parameters(self, recurse: bool = True):
        return self.discriminator.parameters(recurse=recurse)

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse)


class GeneratorNetwork(nn.Module):

    def __init__(self):
        super(GeneratorNetwork, self).__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)


class DiscriminatorNetwork(nn.Module):

    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)


#----------------------------
# DATASET
#----------------------------
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


#----------------------------
# TRAINING + VALIDATION
#----------------------------
def train(model, train_loader, optimizers, device):
    model.train()
    gen_opt, disc_opt = optimizers

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        current_cycle = batch_idx % 2

        if current_cycle == 0:
            # Generator step
            set_requires_grad(model.generator, True)
            set_requires_grad(model.discriminator, False)
            gen_opt.zero_grad()
            gen_loss = model(batch, mode='generator').mean()
            gen_loss.backward()
            gen_opt.step()
        else:
            # Discriminator step
            set_requires_grad(model.generator, False)
            set_requires_grad(model.discriminator, True)
            disc_opt.zero_grad()
            disc_loss = model(batch, mode='discriminator').mean()
            disc_loss.backward()
            disc_opt.step()


def set_requires_grad(model, requires_grad):
    """
    Set the `requires_grad` parameter for every parameter in a model.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def validate(model, val_loader, device):
    model.eval()
    gen_loss_total = 0
    disc_loss_total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            gen_loss = model(batch, mode='generator').mean()
            disc_loss = model(batch, mode='discriminator').mean()
            gen_loss_total += gen_loss.item()
            disc_loss_total += disc_loss.item()

    avg_gen_loss = gen_loss_total / len(val_loader)
    avg_disc_loss = disc_loss_total / len(val_loader)
    print(f'Validation - Generator Loss: {avg_gen_loss}, Discriminator Loss: {avg_disc_loss}')


# Main Function
def main():
    # Initialize the process group for FSDP
    dist.init_process_group(backend='nccl')

    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')

        train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
        val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

        device = 'cuda:0'
        model = AdversarialModel().to(device)

        # Wrap the model with FSDP
        model = FSDP(model)
        gen_opt = optim.SGD(model.generator_parameters(), lr=0.1)
        disc_opt = optim.SGD(model.discriminator_parameters(), lr=0.1)

        num_epochs = 1

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train(model, train_data, (gen_opt, disc_opt), device)
            validate(model, val_data, device)

    # Clean up
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

model.generator = FSDP(model.generator)
model.discriminator = FSDP(model.discriminator)
# do not call `model = FSDP(model)`

import os
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


#----------------------------
# MODEL
#----------------------------
class AdversarialModel(nn.Module):

    def __init__(self):
        super(AdversarialModel, self).__init__()
        self.generator = GeneratorNetwork()
        self.discriminator = DiscriminatorNetwork()

    def forward(self, x, mode='generator'):
        if mode == 'generator':
            return self.generator(x)
        elif mode == 'discriminator':
            return self.discriminator(x)

    def generator_parameters(self, recurse: bool = True):
        return self.generator.parameters(recurse=recurse)

    def discriminator_parameters(self, recurse: bool = True):
        return self.discriminator.parameters(recurse=recurse)

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse)


class GeneratorNetwork(nn.Module):

    def __init__(self):
        super(GeneratorNetwork, self).__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)


class DiscriminatorNetwork(nn.Module):

    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse=recurse)


#----------------------------
# DATASET
#----------------------------
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


#----------------------------
# TRAINING + VALIDATION
#----------------------------
def train(model, train_loader, optimizers, device):
    model.train()
    gen_opt, disc_opt = optimizers

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        current_cycle = batch_idx % 2

        if current_cycle == 0:
            # Generator step
            set_requires_grad(model.generator, True)
            set_requires_grad(model.discriminator, False)
            gen_opt.zero_grad()
            gen_loss = model(batch, mode='generator').mean()
            gen_loss.backward()
            gen_opt.step()
        else:
            # Discriminator step
            set_requires_grad(model.generator, False)
            set_requires_grad(model.discriminator, True)
            disc_opt.zero_grad()
            disc_loss = model(batch, mode='discriminator').mean()
            disc_loss.backward()
            disc_opt.step()


def set_requires_grad(model, requires_grad):
    """
    Set the `requires_grad` parameter for every parameter in a model.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def validate(model, val_loader, device):
    model.eval()
    gen_loss_total = 0
    disc_loss_total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            gen_loss = model(batch, mode='generator').mean()
            disc_loss = model(batch, mode='discriminator').mean()
            gen_loss_total += gen_loss.item()
            disc_loss_total += disc_loss.item()

    avg_gen_loss = gen_loss_total / len(val_loader)
    avg_disc_loss = disc_loss_total / len(val_loader)
    print(f'Validation - Generator Loss: {avg_gen_loss}, Discriminator Loss: {avg_disc_loss}')


# Main Function
def main():
    # Initialize the process group for FSDP
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')

        train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
        val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

        device = 'cuda:0'
        model = AdversarialModel().to(device)

        # Wrap the model with FSDP
        model.generator = FSDP(model.generator)
        model.discriminator = FSDP(model.discriminator)
        gen_opt = optim.SGD(model.generator.parameters(), lr=0.1)
        disc_opt = optim.SGD(model.discriminator.parameters(), lr=0.1)

        num_epochs = 1

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train(model, train_data, (gen_opt, disc_opt), device)
            validate(model, val_data, device)

    # Clean up
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()