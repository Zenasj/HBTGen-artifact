import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.register_buffer("my_val", torch.tensor(0.0))
    
    def forward(self, x):
        return x


model = MyModel()

state_dict = model.state_dict()

print("original model.my_val", model.my_val)
print("original data_ptr", model.my_val.data_ptr(), model.my_val.storage_offset())
print("original state_dict data_ptr", state_dict["my_val"].data_ptr(), state_dict["my_val"].storage_offset())


new_val = torch.rand(10)
model.my_val.resize_(new_val.shape)
model.my_val.copy_(new_val)

print("new model.my_val", model.my_val)
print("new model.my_val ptr", model.my_val.data_ptr(), model.my_val.storage_offset())

print("new data_ptr in state_dict", state_dict["my_val"].data_ptr(), state_dict["my_val"].storage_offset())
print("new state_dict", state_dict)

...
state_dict = model.state_dict(keep_vars=True)
...