import torch
results = dict()
input_tensor = torch.tensor([[0.5786, 0.1719, 0.3760, 0.2939, 0.3984],
        [0.5361, 0.7104, 0.8765, 0.0903, 0.0483]], dtype=torch.float16)
results["res_1"] = torch.std_mean(input_tensor.clone()) # same for var_mean
results["res_2"] = input_tensor.clone().mean()
print(results)
# {'res_1': (tensor(0.2700, dtype=torch.float16), tensor(0.4080, dtype=torch.float16)), 'res_2': tensor(0.4082, dtype=torch.float16)}

import torch
results = dict()
input_tensor = torch.tensor([[0.5078, 0.7773, 0.6836, 0.3438, 0.3672],
        [0.0352, 0.5742, 0.7266, 0.7656, 0.7422]], dtype=torch.bfloat16)
results["res_1"] = torch.std_mean(input_tensor.clone()) # same for var_mean
results["res_2"] = input_tensor.clone().mean()
print(results)
# {'res_1': (tensor(0.2422, dtype=torch.bfloat16), tensor(0.5508, dtype=torch.bfloat16)), 'res_2': tensor(0.5547, dtype=torch.bfloat16)}