import torch.nn as nn

import torch
from functools import partial
from numpy import asarray

class SomeFuns(torch.nn.Module):
    def __init__(self):
        super().__init__()


    @torch.jit.export
    def predict(self, X, batch_size):

        # Build DataLoader
        data =  torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size, False)
        # Batch prediction
        self.eval()
        r, n = 0, X.size()[0]
        for batch_data in data:
            # Predict on batch
            X_batch = torch.autograd.Variable(batch_data[0])
            y_batch_pred = self(X_batch).data
            # Infer prediction shape
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
            # Add to prediction tensor
            y_pred[r : min(n, r + batch_size)] = y_batch_pred
            r += batch_size
        return y_pred




class nonLineaRegression(SomeFuns):
    
    def __init__(self):
        super(nonLineaRegression,self).__init__()
        
        
        self.activation = torch.nn.ReLU()
        
        
        # Input layer
        self.input = torch.nn.Linear(2, 2) 
        
        
        # Hidden layer
        self.hiddenMinusLast = torch.nn.ModuleList()         
        
        layer = torch.nn.Linear(2, 2) 
        self.hiddenMinusLast.append(layer)              
            
        self.hiddenLast = torch.nn.Linear(2, 2) 

            
        # Last layer
        self.output = torch.nn.Linear(2,2) 

    

        
    def forward(self,x):
        
        out = self.activation(self.input(x))
        
        for hiddenLayer in self.hiddenMinusLast:               
            out = self.activation(hiddenLayer(out))
            
        out = self.activation(self.hiddenLast(out))
        out = self.output(out)
        
        return out
 
def main():
    
    model = nonLineaRegression()
  
    script =  torch.jit.script(model)
    torch.jit.save(script,"dbg.pt")
    
if __name__ == '__main__':
    main()