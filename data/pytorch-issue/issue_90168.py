#lstm_test.py
import torch
import torch.nn as nn

torch.use_deterministic_algorithms(True)
seed = 1

for dropout in [0.5, 0]:
    model = nn.LSTM(4, 4, 2, dropout=dropout).cuda()
    model.train()

    for i in range(3):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        data = torch.randn(4, 4)
        if i > 0:
            print(torch.equal(data, pre_data))
        pre_data = data
        data = data.cuda()
        out, _ = model(data)
        loss = out.sum()
        
        print(f"D{dropout}, {i}, Loss: {loss.item()}")