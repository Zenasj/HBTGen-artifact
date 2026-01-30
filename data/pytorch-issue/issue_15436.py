import torch
import torch.nn as nn

X, y = make_classification(n_samples=60000,n_features=10, n_redundant=0, n_informative=2)

X_train,y_train, X_test, y_test = train_test_split(X, y, stratify=y, random_state=42)
X_train = Variable(torch.from_numpy(X_train)).double()
y_train = Variable(torch.from_numpy(y_train)).double()
X_test = Variable(torch.from_numpy(X_test)).double()
y_test = Variable(torch.from_numpy(y_test)).double()

input_dim = 40000
output_dim = 2
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
batch_size = 100
n_iters = 3000
num_epochs = int(n_iters / (len(X_train) / batch_size))

model = SingleLayeredNetwork(input_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

class SingleLayeredNetwork(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(SingleLayeredNetwork, self).__init__()
        self.layer = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        out = self.layer(x)
        return out


iter = 0
for epoch in range(num_epochs):
    for train, test in zip(X_train, y_train):
        print(train.dtype)
        print(torch.typename(train))
        optimizer.zero_grad()
        
        outputs = model(train)
        loss = criterion(outputs, test)
        
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for tra, tes in zip(X_test, y_test):
                # Load images to a Torch Variable
                tra = tra

                # Forward pass only to get logits/output
                outputs = model(train)

                # Get predictions from the maximum value
                # 100 x 1
                _, predicted = torch.max(train, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == test).sum()

            accuracy = 100 * correct.item() / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))