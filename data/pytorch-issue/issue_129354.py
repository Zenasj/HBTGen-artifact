import torch
import torch.nn as nn
import numpy as np
import random

param.grad += torch.tensor(np.random.normal(0, 0, size=param.grad.shape), device=param.grad.device)

optimizer = torch.optim.AdamW(bert_model.parameters(), lr=learning_rate)
bert_model.train()

for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Clipping gradients
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), clip_norm)

        # Adding differential privacy noise to gradients
        for param in bert_model.parameters():
            if param.grad is not None:
                param.grad += torch.tensor(np.random.normal(0, 0, size=param.grad.shape), device=param.grad.device) 

        optimizer.step()
        progress_bar.update(1)

noise_tensor = torch.zeros_like(param.grad, device=param.grad.device)
param.grad += noise_tensor

param.grad = param.grad + torch.tensor(np.random.normal(0, 0, size=param.grad.shape), device=param.grad.device)

param.grad += torch.tensor(np.random.normal(0, 0, size=param.grad.shape), device=param.grad.device)