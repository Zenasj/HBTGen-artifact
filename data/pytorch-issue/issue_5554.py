import torch.nn as nn

criterion = nn.CrossEntropyLoss()

print(outputs.data)
print(label.data)
loss = criterion(outputs, label)      # getting error at this point