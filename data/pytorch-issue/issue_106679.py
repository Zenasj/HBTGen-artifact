import torch
import torch.nn as nn

class NonLinearRegModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers: list = [5000, 1000, 500, 100], dropout: float = 0.5, activation: str = 'relu'):
        super(NonLinearRegModel, self).__init__()

        # List of activation functions
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'prelu': nn.PReLU()
        }
        
        # List to hold the layers
        self.layers = nn.ModuleList()
        
        # Initial First Layer
        # Layer: NonLinear -> BatchNorm -> Activation -> Dropout
        self.layers.append(nn.Linear(in_features, hidden_layers[0]))
        #self.layers.append(nn.LayerNorm(hidden_layers[0]))  # Rather than using BatchNorm1d as this normalises across the batch dimension, LayerNorm normalises across the feature dimension
        self.layers.append(nn.BatchNorm1d(hidden_layers[0]))
        self.layers.append(activation_functions[activation])
        self.layers.append(nn.Dropout(dropout))
        
        # Constructing intermediate hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            self.layers.append(activation_functions[activation])
            self.layers.append(nn.Dropout(dropout))
        
        # Final output layer
        self.layers.append(nn.Linear(hidden_layers[-1], out_features))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Training Loop

# Init lists to store the losses and METRICS
train_losses = []
test_losses = []
train_maes = []
test_maes = []

train_mses = []
test_mses = []

train_r2s = []
test_r2s = []

for epoch in range(epochs):
    
    
    ### Training
    model_0.train()
    
    # 1. Forward Propagation
    y_pred = model_0(X_train)
    
    # 2. Loss Calculation
    loss = log_cosh_loss(y_pred, y_train)
    train_losses.append(loss.item())
    train_pred_np = y_pred.detach().cpu().numpy()
    train_true_np = y_train.detach().cpu().numpy()

    train_maes.append(mean_absolute_error(train_true_np, train_pred_np))
    train_mses.append(mean_squared_error(train_true_np, train_pred_np))
    train_r2s.append(r2_score(train_true_np, train_pred_np))
    
    # 3. Zero Grad Optimization
    optimizer.zero_grad()
    
    # 4. Backward Propagation
    loss.backward()
    
    # 5. Step Optimization
    optimizer.step()
    
    ### Validation
    model_0.eval() # set model to evaluation mode
    # 1. Forward Propagation
    with torch.no_grad():
        y_pred_test = model_0(X_test)
        
        # 2. Loss Calculation
        test_loss = log_cosh_loss(y_pred_test, y_test)
        test_losses.append(test_loss.item())  # Store test loss for this epoch
        test_pred_np = y_pred_test.detach().cpu().numpy()
        test_true_np = y_test.detach().cpu().numpy()

        test_maes.append(mean_absolute_error(test_true_np, test_pred_np))
        test_mses.append(mean_squared_error(test_true_np, test_pred_np))
        test_r2s.append(r2_score(test_true_np, test_pred_np))

    # Scheduler Step
    scheduler.step(test_loss)

    if epoch % 500 == 0:
        print(f'Epoch {epoch} | Train Loss: {train_losses[-1]} | Test Loss: {test_losses[-1]}')
        print(f'Train MAE: {train_maes[-1]} | Test MAE: {test_maes[-1]}')
        print(f'Train MSE: {train_mses[-1]} | Test MSE: {test_mses[-1]}')
        print(f'Train R^2: {train_r2s[-1]} | Test R^2: {test_r2s[-1]}')
        print("---------------------------------------------------")