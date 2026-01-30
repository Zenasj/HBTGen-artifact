class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,x):
        return self.model(x.permute(0,3,1,2))