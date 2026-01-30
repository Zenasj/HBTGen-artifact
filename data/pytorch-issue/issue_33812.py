import numpy as np
import torch
numpy_matrix = np.empty((128, 128, 3))
numpy_matrix = numpy_matrix.transpose((2, 0, 1))[np.newaxis]
torch_tensor = torch.from_numpy(numpy_matrix).cuda(device=0)