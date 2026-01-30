from flask import Flask
import torch
import torch.nn as nn 
app = Flask()
app.run(port=8000, processes=4)


class SimpleLinearModel(nn.Module):
    def __init__(self):
          super(SimpleLinearModel, self).__init__()
          self.linear_layer = nn.Linear(2500, 300)
    
    def forward(x):
         return self.linear_layer(x)

model = SimpleLinearModel()
model(torch.randn(1, 2500))

from flask import Flask
import torch
import torch.nn as nn 
app = Flask()
app.run(port=8000, processes=4)

class SimpleLinearModel(nn.Module):
    def __init__(self):
          super(SimpleLinearModel, self).__init__()
          self.linear_layer = nn.Linear(2500, 300)
    
    def forward(x):
         return self.linear_layer(x)


@app.route("/run_model", methods=['POST'])
def run_model():
      model = SimpleLinearModel()
      model(torch.randn(1, 2500))