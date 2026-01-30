import torch

def convert_model(model, input=torch.tensor(torch.rand(size=(1,3,112,112)))):
        model = torch.jit.trace(self.model, input)
        torch.jit.save(model,'/home/Rika/Documents/models/model.tjm')

# load the model 
self.model = torch.jit.load('/home/Rika/Documents/models/model.tjm')