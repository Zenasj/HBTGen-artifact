import torch.nn as nn
import torchvision
import numpy as np

class ModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs=30,
        patience=5,
        optimizer=None,
        criterion=nn.CrossEntropyLoss(),
        scheduler_step_size=10,
        scheduler_gamma=0.1,
    ):
        self.device = self._get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience
        ## It would be more efficient to provide the model with the head parameters
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=1e-3)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )
        self.criterion = criterion
        ## History
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _get_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        return device

    def train(self):

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_weights = None

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                    dataloader = self.train_loader
                else:
                    self.model.eval()
                    dataloader = self.val_loader

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloader, desc=phase, unit=" batch"):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)

                # Calculation adjusted for architecture
                if self.device == "cuda" or self.device == "cpu":
                    epoch_acc = running_corrects.double() / len(dataloader.dataset)
                elif self.device == "mps":  # Assuming Metal (MPS) backend
                    epoch_acc = running_corrects.float() / len(dataloader.dataset)
                else:  # For TPU or any other backend
                    epoch_acc = running_corrects.to(torch.float64) / len(
                        dataloader.dataset
                    )

                print(f"\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                print()
                print()

                if phase == "train":
                    self.train_loss_history.append(epoch_loss)
                    self.train_acc_history.append(epoch_acc)
                else:
                    self.val_loss_history.append(epoch_loss)
                    self.val_acc_history.append(epoch_acc)

                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_weights = self.model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1

            print()

            if patience_counter >= self.patience:
                print(
                    f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss."
                )
                break

        print("Training complete.")
        self.model.load_state_dict(best_model_weights)
        return self.model

    def history(self):
        # Move data to CPU and convert to NumPy arrays
        train_loss_history_np = np.array(self.train_loss_history)
        val_loss_history_np = np.array(self.val_loss_history)
        train_acc_history_np = [acc.cpu().numpy() for acc in self.train_acc_history]
        val_acc_history_np = [acc.cpu().numpy() for acc in self.val_acc_history]

        # Plot loss and accuracy history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss history
        axes[0].plot(train_loss_history_np, label="Train Loss", color="blue")
        axes[0].plot(val_loss_history_np, label="Validation Loss", color="orange")
        axes[0].set_title("Loss History")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Plot accuracy history
        axes[1].plot(train_acc_history_np, label="Train Accuracy", color="blue")
        axes[1].plot(val_acc_history_np, label="Validation Accuracy", color="orange")
        axes[1].set_title("Accuracy History")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.show()

        return {
            "train_loss": train_loss_history_np,
            "val_loss": val_loss_history_np,
            "train_acc": train_acc_history_np,
            "val_acc": val_acc_history_np,
        }

import torch
from torchvision.models.resnet import resnet50

def main() -> None:
    model = resnet50()
    for name, param in model.named_parameters():
        if "bn" in name or "batchnorm" in name.lower():
            param.requires_grad = False

    model.to(device='mps')
    inputs = torch.rand(1, 3, 224, 224, device='mps')
    outputs = model(inputs)
    outputs.sum().backward()

if __name__ == "__main__":
    main()

import torch
inputs = torch.rand(1, 8, 4, 4, device='mps', requires_grad=True)
x = torch.nn.BatchNorm2d(8).to("mps")
y = torch.nn.BatchNorm2d(8).to("mps")
y.weight.requires_grad=False
y.bias.requires_grad=False
outputs = y(x(inputs))
outputs.sum().backward()