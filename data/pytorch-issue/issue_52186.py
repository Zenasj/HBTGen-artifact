import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------
# load the dataset
train, test, feature_num = loading_data(data_root_path, combine_feature, n_hour)

# assign to dataloader    
train_data = click_dataset(train)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM_model(input_size=feature_num, hidden_size=hidden_size, num_layers=num_layers).double().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# manually set an early-stopping
best_val_loss = 99999
delta = 0.0001
patience = 30

# loop the dataset
for epoch in range(epoch_num): 

    t1 = time.time()

    for _, data in enumerate(train_dataloader, 0):

        # get the inputs; data is a list of [features, QTY]
        inputs = data[0].to(device)
        labels = data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        training_loss = criterion(outputs, labels)
        training_loss.backward()
        optimizer.step()
    
    with torch.no_grad():

        # get the inputs; data is a list of [features, QTY]
        val_inputs = torch.from_numpy(test[0]).double().to(device)
        val_labels = torch.from_numpy(test[1]).double().to(device)

        # forward + backward + optimize
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)

        # Early stopping
        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            wait = 0
            torch.save(model, model_save_path + model_name+ "_exp_" + exp_number)

            # print statistics
            print('\n [%d] Train loss: %.6f' % (epoch + 1, training_loss), \
                ', Val loss: %.6f' % (val_loss.item()), \
                    ', time: ', round(time.time() - t1, 3), 's, ', \
                    " --- model saved!")
            
        else:
            if wait >= patience:
                print('\n Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

            # print statistics
            print('\n [%d] Train loss: %.6f' % (epoch + 1, training_loss), \
                ', Val loss: %.6f' % (val_loss.item()), ', time: ', round(time.time() - t1, 3), 's')

# now test on test dataset
model = torch.load(model_save_path + model_name+ "_exp_" + exp_number)
model.eval()

with torch.no_grad():
    # now do the prediction
    test_inputs = torch.from_numpy(test[0]).double()
    test_outputs = model(test_inputs)
    test_labels = torch.from_numpy(test[1]).double()

    test_loss = criterion(test_outputs, test_labels)
    print("\ntest_loss: {:.6f} \n".format(test_loss))

    test_loss = np.abs((test_outputs - test_labels).numpy())
    test_result = np.mean(np.square(test_loss.reshape(-1,1000,1)),axis=1)
    print("\ntest_result: ", [round(i.item(),5) for i in test_result])

class LSTM_model(nn.Module):

    def __init__(self, input_size, hidden_size=16, num_layers=2):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # x: (batch_size, seq, fea_number/input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 1)
        
    def forward(self, x):

        # Set initial hidden states
        # x: (batch_size, time_length, fea_number)
        # h0: (num_layers * num_directions, batch_size, hidden_size)
        # c0: (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).double()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).double()
          
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)        
        # Only need the hidden state of the last time step
        # out: (batch_size, hidden_size)
        out = out[:, -1, :]

        # out: (batch_size, 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out