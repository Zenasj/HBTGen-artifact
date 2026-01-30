import random

import numpy as np                                                                          
import torch                                                                                       

def sample(inputs: torch.tensor, num_samples: int):                                       
    samples = np.random.choice(inputs.shape[0], num_samples) 
    inputs = inputs[samples]                                                              
    return inputs                                                                                  
                                   
# np_inputs is of type np.ndarray with shape (-1, 2, 384)                                                                    

inputs = torch.tensor(np_inputs, dtype=torch.float, device=torch.device('mps')).view(-1, 1, 2, 384)

inputs = sample(inputs, num_samples)