import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module) :
    def __init__(self,index_info) :
          ...
    def forward(self,x ):
          ...

class RegressorModule(nn.Module):
    def __init__(self,num_units_1=80,num_units_2 = 60, num_units_3 = 20, nonlin=nn.Relu(),index_info=index_info):
        super(RegressorModule, self).__init__()
        self.index_info = index_info
        if self.index_info is not None :
            self.emb_layer = EmbeddingLayer(self.index_info)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.1)
        self.dense0 = nn.Linear(self.emb_layer.output_n, num_units_1)
        self.dense1 = nn.Linear(num_units_1, num_units_2)
        self.dense2 = nn.Linear(num_units_2, num_units_3)
        self.output = nn.Linear(num_units_3, 1)

    def forward(self, X):
        if self.index_info is not None :
            X = self.emb_layer(X)
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        X = self.output(X)
        return X
pipe = RegressorModule()

print("save model...")              
m = torch.jit.script(pipe)
with torch.no_grad() :
    m.eval()
    torch.jit.save(m,  os.path.join(model_path ,'freeze_model.pt'))