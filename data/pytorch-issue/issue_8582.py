for name, param in model.named_parameters():
    if name == 'encoder.embedding.weight':
        param.requires_grad = False
loss = model.forward()
loss.backward()

for name, param in model.named_parameters():
    if name == 'encoder.embedding.weight':
        param.requires_grad = False
model.train()
loss = model.forward()
loss.backward()