import torch
import torch.nn as nn

class CustomModelWithGradCAM(torch.nn.Module):
    def __init__(self, model_weights, 
                 model_device) -> None:
        super(CustomModelWithGradCAM, self).__init__()  
        
        self.model = CustomModel() # The same as code above already trained
        self.model.load_state_dict(torch.load(model_weights)) # Load its weights
        self.model.to(model_device)
        self.model.eval()
    
        self.model_device = model_device
        
    def getCAM(predictions, classId):
        predictions[:, classId].backward()
        # pull the gradients out of the model
        gradients = self.model.get_activations_gradient()
        ## etc etc etc etc...
     
    def forward(self, x):
        outputs = self.model(x)
        heatmaps = self.getCAM(predictions=outputs, classId=1)
        ## etc etc etc etc...