import torch.nn as nn

import torch.nn.functional as F
proportion = int(0.8*len(x)) 
x_train, y_train = x[:proportion], y[:proportion]
x_test, y_test = x[proportion:], y[proportion:] 

BATCH_SIZE = 256
train_dataset = TensorDataset(torch.LongTensor(x_train.type(torch.int64)), torch.LongTensor(y_train.type(torch.int64)))
train_dataloader = DataLoader(dataset=train_dataset,
                           batch_size=BATCH_SIZE, 
                           shuffle=True)
test_dataset = TensorDataset(torch.LongTensor(x_test.type(torch.int64)), torch.LongTensor(y_test.type(torch.int64)))
test_dataloader = DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

#Create the neural network 
class incomemodel(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.embedded_layer = nn.Sequential(
            nn.Embedding(num_embeddings=4, embedding_dim=8), 
            nn.Embedding(num_embeddings=4, embedding_dim=8), 
            nn.Embedding(num_embeddings=4, embedding_dim=8)
        )
        self.stacked_layer = ( 
            nn.Linear(in_features=24, out_features=1052),
            nn.ReLU(), 
            nn.BatchNorm1d(1052),
            nn.Linear(in_features=1052, out_features=526), 
            nn.ReLU(), 
            nn.BatchNorm1d(526),
            nn.Linear(in_features=526, out_features=256), 
            nn.ReLU(), 
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=4),
        )
    
    def forward(self, x): 
        print("Input data type:", x.dtype)
        embedded_features = self.embedded_layer(x).type(torch.float32)
        print("Embedded features data type:", embedded_features.dtype)
        return self.stacked_layer(embedded_features)

#Initialize the model
model = incomemodel() 
loss_fn = nn.KLDivLoss(reduction='batchmean') 
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
def acc_fn(y_pred, y_true): 
    correct = torch.eq(y_pred, y_true).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc 

#Create the tranining and testing model 
epoches = 200
loss_lis = []
test_acc_list = []
for epoch in range(epoches): 
    print(f'Epoch at {epoch}\n---')
    for batch, (train_features, train_labels) in enumerate(train_dataloader): 
        train_labels = train_labels.long()
        # train_labels = F.one_hot(train_labels, num_classes=4)
        #1. Pass to the forward 
        train_features = train_features.view(-1, 12)
        print(train_features.dtype)
        train_pred = model(train_features)
        train_log = F.log_softmax(train_pred, dim=1)

        #2. Calculate the loss 
        train_loss = loss_fn(train_log, train_labels.type(torch.float32))

        #3. Backward propagation on the loss 
        train_loss.backward()

        #4. Gradient descending 
        optimizer.step() 

        #5. Zero out the gradien 
        optimizer.zero_grad() 

        if batch % 40 == 0: 
            print(f'Looked at {batch * len(train_features)}/{len(train_dataloader.dataset)} samples')

    #Test the model
    model.eval() 
    with torch.inference_mode(): 
        for test_features, test_labels in test_dataloader:
            test_labels = test_labels.long()
            # test_labels = F.one_hot(test_labels, num_classes=4)
            #1. Pass to the forward
            test_pred = model(test_features.type(torch.long))
            log_test = F.log_softmax(test_pred, dim=1)

            #2. Calculate the loss 
            test_loss = loss_fn(log_test, test_labels.type(torch.float32))
            test_acc = acc_fn(y_pred=torch.argmax(test_pred, dim=1),
                                y_true=test_labels.argmax(dim=1))
            
            test_acc_list.append(test_acc)
            loss_lis.append(train_loss.item())
            
            if test_acc == 100: 
                break 
            
    print(f'Train loss {train_loss.item():.4f} | Test loss {test_loss.item():.4f} | Test acc {test_acc}')