import torch 
import torch.nn as nn 

pool = nn.MaxPool1d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool1d(2, stride=2) 
 
# Example showcasing the use of output_size
input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
output, indices = pool(input)
unpool(output, indices, output_size=input.size())