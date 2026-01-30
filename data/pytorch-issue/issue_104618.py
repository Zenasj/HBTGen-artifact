# result same for AdaptiveAvgPool2d and AdaptiveAvgPool3d
import torch.nn as nn
input_sizes_1d =[1,2] 
m = nn.AdaptiveAvgPool1d(0)
m(torch.rand(input_sizes_1d))

m = nn.AdaptiveAvgPool1d(0)
torch.compile(m)(torch.rand([1,2]))  # Raises LoweringException

m = nn.AdaptiveAvgPool2d(0)
torch.compile(m)(torch.rand([1,2,3]))  # Raises LoweringException

m = nn.AdaptiveAvgPool3d(0)
torch.compile(m)(torch.rand([1,2,3,4]))  # Using FallbackKernel: aten._adaptive_avg_pool3d

import torch.nn as nn
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.AdaptiveAvgPool1d(output_size=0)
    def forward(self, inputs):

        return self.layer(inputs)
    
input_tensor = torch.randn([1,2]) 
# Create the model
mymodel = CustomModel()

# Forward pass
output = mymodel(input_tensor) ## No error
mymodel.to('cuda')
op_output = torch.compile(mymodel)(input_tensor)