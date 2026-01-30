import numpy as np
import torch


def compare_bn_layers(bn1_state_dict, bn2_state_dict):
    for key in bn1_state_dict.keys():

        if "running_mean" in key or "running_var" in key:
            continue

        if not torch.all(torch.eq(bn1_state_dict[key], bn2_state_dict[key])):
            return False

    return True


new_layer_path = "./new_Conv2d.pth"
new_data_path = "./data0_Conv2d.npy"
new_layer = torch.load(new_layer_path)
new_data = np.load(new_data_path)

old_layer_path = "./old_Conv2d.pth"
old_data_path = "./data0_Conv2d.npy"
old_layer = torch.load(old_layer_path)
old_data = np.load(old_data_path)

new_layer.eval()
old_layer.eval()
input_data_new = torch.from_numpy(new_data)
input_data_old = torch.from_numpy(old_data)

output_new = new_layer(input_data_new)
output_old = old_layer(input_data_old)

# Ensure there is no difference in input
print(np.max(np.abs(input_data_new.detach().cpu().numpy() - input_data_old.detach().cpu().numpy())))
# Measure the difference in output
print(np.max(np.abs(output_new.detach().cpu().numpy() - output_old.detach().cpu().numpy())))
# Determine weight
print(compare_bn_layers(new_layer.state_dict(), old_layer.state_dict()))