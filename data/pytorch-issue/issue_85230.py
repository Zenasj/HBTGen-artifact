import torch
import requests
# Download checkpoint file
ckpt='FastSurferVINN_training_state_coronal.pkl'
fileurl='https://b2share.fz-juelich.de/api/files/0114331a-f788-48d2-9d09-f85d7494ed48/FastSurferVINN_training_state_coronal.pkl'
response = requests.get(fileurl, verify=False)
with open(ckpt, 'wb') as f:
    f.write(response.content)

# CPU load works:
model_state = torch.load(ckpt, map_location="cpu")
print(model_state["model_state"]["inp_block.bn0.weight"])
# ouput: tensor([2.0432, 1.2577, 4.1133, 7.4062, 3.9921, 1.8011, 2.0956])

# MPS load gives zeros:
model_state = torch.load(ckpt, map_location="mps")
print(model_state["model_state"]["inp_block.bn0.weight"])
#output tensor([0., 0., 0., 0., 0., 0., 0.], device='mps:0')