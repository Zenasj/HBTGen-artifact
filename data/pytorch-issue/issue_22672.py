import torch

with torch.no_grad():
        outputs = model(img)
        outputs = torch.sigmoid(outputs)
        score = outputs[:,0,:,:]
        outputs = outputs > 0.7