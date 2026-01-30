import torch
import numpy as np

cpu_device=torch.device('cpu')

# load trained model
model=torch.jit.load(model_loc, map_location=cpu_device)

inp=torch.ones([3,112,112]).unsqueeze(0)

output = model(inp)

print(output.shape)

np.savetxt("run05_pfc.out", output.detach().numpy(), delimiter="\t")

sys.exit()