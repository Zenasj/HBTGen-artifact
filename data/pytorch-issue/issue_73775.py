import torch.nn as nn

batch_size, time_steps = 256, 35
train_iter, vocab = load_data_novel(batch_size, time_steps)
num_hiddens, num_layers = 256, 1
lstm_layer = nn.GRU(len(vocab), num_hiddens)
model = RNNModel(lstm_layer, len(vocab))
device, lr, num_epochs = d2l.try_gpu(), 1, 500
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0, 1])