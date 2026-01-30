import torch 
import torch.nn as nn 
import torch.distributed.checkpoint as dcp 

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

model = ToyModel()

state_dict = {"model": model.state_dict()}

# Change to this a path you have write access to 
path = "s3://<path to folder>" # or GCS path 

dcp.save(state_dict, checkpoint_id=path)