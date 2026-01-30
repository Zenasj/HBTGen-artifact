py
import torch.nn as nn

class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def property_a(self):
        return self.property_b

m = CustomModule()
print(m.property_a)

py
class CustomModule2:
    def __init__(self):
        pass

    @property
    def property_a(self):
        return self.property_b

m2 = CustomModule2()
print(m2.property_a)