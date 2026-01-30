import torch.nn as nn

# import dependencies
import os
import json
import torch
import numpy as np

layers = [100, 300, 200, 5]

# create model
class model(torch.nn.Module):
    """
    Neural Network Class
    Parameters
    ----------
    input : int
        number of input features
    neurons : int
        number of units in the hidden layer
    Returns
    -------
    tensor: torch.Tensor
        output tensor of the neural network
    """
    def __init__(self, input: int, hidden_sizes: list, p: float = 0.2, device: str = 'cuda') -> None:
        """
        Constructor
        Parameters
        ----------
        input : int
            number of input features
        hidden_sizes: list
            list of the number of neurons in each hidden layer
        p : float
            dropout probability
        device : str
            device to train the model on
        
        Returns
        -------
        None
        """
        super().__init__()

        self.device = torch.device(device)
        self.neural_network = torch.nn.ModuleList()
        sizes = [input] + hidden_sizes + [1]
        for i in range(1, len(sizes)):
            self.neural_network.append(torch.nn.Linear(sizes[i-1], sizes[i]))
            if i < len(sizes) - 1:
                if i != len(sizes) - 1:
                    self.neural_network.append(torch.nn.ReLU())
                    # self.neural_network.append(torch.nn.Dropout(p=p))


    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Parameters
        ----------
        tensor : torch.Tensor
            input tensor
        
        Returns
        -------
        tensor: torch.Tensor
            output tensor of the neural network
        """
        for layer in self.neural_network:
            tensor = layer(tensor.to(self.device))

        return tensor
    
    def train_model(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, 
                    y_test: np.ndarray, epochs: int, batch_size: int, 
                    creterian: torch.nn.modules.loss, optimizer: torch.optim, path: str) -> None:
        """
        Train the neural network
        Parameters
        ----------
        x_train : np.ndarray
            training features
        y_train : np.ndarray
            training labels
        x_test : np.ndarray
            testing features
        y_test : np.ndarray
            testing labels
        epochs : int
            number of epochs
        batch_size : int
            batch size
        creterian : torch.nn..modules.loss
            loss function
        optimizer : torch.optim
            optimizer
        path : str
            path to save the training log
        Returns
        -------
        None
        """

        # assert the arguments data types
        assert isinstance(x_train, np.ndarray), 'x_train must be a numpy array'
        assert isinstance(y_train, np.ndarray), 'y_train must be a numpy array'
        assert isinstance(x_test, np.ndarray), 'x_test must be a numpy array'
        assert isinstance(y_test, np.ndarray), 'y_test must be a numpy array'
        assert isinstance(epochs, int), 'epochs must be an integer'
        assert isinstance(batch_size, int), 'batch_size must be an integer'
        assert "torch.nn.modules.loss" in str(type(creterian)), 'creterian must be a torch.nn.modules.loss'
        assert 'torch.optim' in str(type(optimizer)), 'optimizer must be a torch.optim'
        assert isinstance(path, str), 'path must be a string'

        # training log file
        log_train = os.path.join(path, 'training.log')
        json_losses = os.path.join(path, 'losses.json')

        # training losses
        train_losses = []

        # validation log file
        log_val = os.path.join(path, 'losses.log')

        # validation losses
        val_losses = []

        # train model
        status = os.path.exists(log_train)
        if status:
            train_mode = 'w'
        else:
            train_mode = 'x'
        
        status = os.path.exists(log_val)
        if status:
            val_mode = 'w'
        else:
            val_mode = 'x'
        
        with open(log_train, train_mode) as train_file, open(log_val, val_mode) as val_file:
            for epoch in range(epochs):
                self.train()
                for i in range(0, x_train.shape[0], batch_size):
                    x_batch = torch.tensor(x_train[i:i+batch_size], device=self.device, dtype=torch.float)
                    y_batch = torch.tensor(np.array(y_train[i:i+batch_size]).reshape((y_train[i:i+batch_size].shape[0], 1)), device=self.device, dtype=torch.float)
                    y_pred = self.forward(x_batch).to(self.device)
                    loss = creterian(y_pred, y_batch).float()
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
                    optimizer.step()
                    train_file.write(f'Epoch: {epoch + 1}, Batch: {int(i/batch_size)}, Training Loss: {loss.item()}\n')

                with torch.no_grad():
                    self.eval()
                    x = torch.tensor(x_train, device=self.device).float()
                    y = torch.tensor(np.array(y_train).reshape((y_train.shape[0], 1)), device=self.device).float()
                    x_val = torch.tensor(x_test, device=self.device).float()
                    y_val = torch.tensor(np.array(y_test).reshape((y_test.shape[0], 1)), device=self.device).float()
                    y_pred_val = self.forward(x_val).to(self.device)
                    y_pred_train = self.forward(x).to(self.device)
                    train_loss = creterian(y_pred_train, y)
                    train_losses.append(train_loss.item())
                    val_loss = creterian(y_pred_val, y_val)
                    val_losses.append(val_loss.item())
                    val_file.write(f'Epoch: {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss.item()}\n')
        
        with open(json_losses, 'w') as train_file:
            json.dump({'train': train_losses, 'val': val_losses}, train_file)
        
            
    
    def save(self, path: str) -> None:
        """
        Save the model as torch script
        Parameters
        ----------
        path : str
            path to save the model
        Returns
        -------
        None
        """
        
        # assert the arguments data types
        assert isinstance(path, str), 'path must be a string'

        # save model
        torch.jit.save(torch.jit.script(self.float()), path)