import torch.nn as nn

import torch

class Architecture(torch.nn.Module):

    def __init__(self, n_features, n_classes):
        super(Architecture, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes

        self.cls = torch.nn.Linear(self.n_features, self.n_classes)

    def forward(self, x):
        ts, bs = x.shape[:2]

        x = x.view(ts * bs, self.n_features)
        x = self.cls(x).view(ts, bs, self.n_classes)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x

BATCH_SIZE = 2
GT_LENGTH = 3
N_TIMESTEPS = GT_LENGTH * 2
N_FEATURES = 5
N_CLASSES = 4

def get_model():
    arch = Architecture(N_FEATURES, N_CLASSES)
    arch.train()

    return arch

def get_data():
    # Data
    features = torch.normal(mean=torch.zeros((N_TIMESTEPS, BATCH_SIZE, N_FEATURES), dtype=torch.float32))
    pred_lengths = N_TIMESTEPS * torch.ones((BATCH_SIZE,), dtype=torch.int32)

    targets = torch.randint(1, N_CLASSES, size=(BATCH_SIZE * GT_LENGTH,), dtype=torch.int32)
    target_lengths = GT_LENGTH * torch.ones((BATCH_SIZE,), dtype=torch.int32)

    return features, pred_lengths, targets, target_lengths

def cast_data(features, pred_lengths, targets, target_lengths, device, dtype):
    features = features.to(device)
    pred_lengths = pred_lengths.to(dtype)
    targets = targets.to(dtype).to(device)
    target_lengths = target_lengths.to(dtype)

    if dtype == torch.int32:
        targets = targets.to(torch.device("cpu"))

    return features, pred_lengths, targets, target_lengths

def run(model, data, device, dtype, loss_mult=1.0):
    if device == torch.device("cpu"):
        print("\n# CTC CPU     : device {} - dtype {} - mul {}".format(device, dtype, loss_mult))
    elif device == torch.device("cuda") and dtype == torch.int32:
        print("\n# CTC CUDNN   : device {} - dtype {} - mul {}".format(device, dtype, loss_mult))
    elif device == torch.device("cuda") and dtype == torch.long:
        print("\n# CTC REGULAR : device {} - dtype {} - mul {}".format(device, dtype, loss_mult))

    model = model.to(device)
    features, pred_lengths, targets, target_lengths = cast_data(*data, device, dtype)
    preds = model(features)

    # Loss
    loss = torch.nn.functional.ctc_loss(preds, targets, pred_lengths, target_lengths)

    loss = loss_mult * loss

    print('LOSS : ', loss)

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        print('GRAD : ', param.grad.abs().mean())

    model.zero_grad()

data = get_data()
model = get_model()

print("----- CPU -----")
run(model, data, torch.device("cpu"), torch.int32)
run(model, data, torch.device("cpu"), torch.long)
run(model, data, torch.device("cpu"), torch.int32, 0.0)
run(model, data, torch.device("cpu"), torch.long, 0.0)

print("\n----- GPU -----")
run(model, data, torch.device("cuda"), torch.int32)
run(model, data, torch.device("cuda"), torch.long)
run(model, data, torch.device("cuda"), torch.int32, 0.0)
run(model, data, torch.device("cuda"), torch.long, 0.0)