import torch.nn as nn

import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn

# from utils import get_dataloaders

warnings.filterwarnings("ignore")


def conv_block(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.ReLU(),
        nn.Dropout2d(0.25))


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.model_size = 1
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * self.model_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8 * self.model_size, out_channels=16 * self.model_size, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=16 * self.model_size, out_channels=32 * self.model_size, kernel_size=3,
                               padding=1)
        self.conv = nn.Conv2d(in_channels=16 * self.model_size, out_channels=16 * self.model_size, kernel_size=3,
                              padding=1)

        self.conv_block_start = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8 * self.model_size, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.model_size, out_channels=16 * self.model_size, kernel_size=3, stride=2,
                      padding=1), nn.ReLU())
        self.conv_block1 = conv_block(in_channels=16 * self.model_size, out_channels=16 * self.model_size,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)
        self.conv_block2 = conv_block(in_channels=16 * self.model_size, out_channels=16 * self.model_size,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)
        self.global_avg_pooling2D = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Linear(in_features=16, out_features=2)

    def forward(self, t):
        t = t

        t = self.conv_block_start(t)
        t = self.conv_block1(t)
        t = self.conv_block2(t)

        t = self.global_avg_pooling2D(t)
        t = t.view(-1, t.shape[1])

        t = self.out(t)
        t = F.softmax(t, dim=1)
        return t

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_probability = self(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat_probability, y)

        y_hat = torch.argmax(y_hat_probability, dim=1).cpu()
        acc = torch.tensor(accuracy_score(y.cpu(), y_hat, normalize=True))
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_probability = self(x)
        loss_fn = torch.nn.CrossEntropyLoss()

        val_loss = loss_fn(y_hat_probability, y)
        y_hat = torch.argmax(y_hat_probability, dim=1).cpu()
        val_acc = torch.tensor(accuracy_score(y.cpu(), y_hat, normalize=True))
        return {"val_loss": val_loss,
                "val_acc": val_acc}

    def training_epoch_end(self, outputs):
        acc = torch.stack([x["acc"] for x in outputs]).mean()

        log = {"acc": acc}

        return {"log": log, "progress_bar": log}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        log = {"val_loss": val_loss,
               "val_acc": val_acc}

        return {"log": log, "progress_bar": log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    # def train_dataloader(self):
    #     train_dataloader = get_dataloaders()[0]
    #     return train_dataloader
    #
    # def val_dataloader(self):
    #     val_dataloader = get_dataloaders()[1]
    #     return val_dataloader


if __name__ == "__main__":
    rand_tensor = torch.rand(1, 3, 256, 256)
    net = Net()
    test = net(rand_tensor)
    # print(test.shape)
    # summary(net, rand_tensor)
    # train_dataloader, val_dataloader = get_dataloaders()
    # trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    # trainer.fit(net)
    # torch.save(net, "bare_rtf")
    torch.onnx.export(net, rand_tensor.size(), "bare_rtf.onnx", verbose=True)