import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Dropout(p=1, inplace=False)
    def forward(self, inputs):

        return self.layer(inputs)
    
input_tensor = torch.randn([1,2]) 
# Create the model
mymodel = CustomModel()

# Forward pass
output = mymodel(input_tensor) ## No error
mymodel.to('cuda')
op_output = torch.compile(mymodel)(input_tensor)