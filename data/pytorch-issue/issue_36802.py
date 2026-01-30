import torch

for nepoch in range(8):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)

device = torch.device("cuda:0")
for nepoch in range(8):
    device = torch.device("cuda:0")
    train_one_epoch(qat_model, criterion, optimizer, train_loader, torch.device(device), num_train_batches)

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to(device) #device instead of 'cpu'
    return model