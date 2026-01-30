import torch
import torch.nn as nn

class mini_attn(ScriptModule):
    def __init__(self):
        super().__init__()
        self.rnn=trace(nn.LSTMCell(5,10),(torch.rand(1,5),(torch.rand(1,10),torch.rand(1,10))))
    
    @script_method
    def forward(self,input):

        hx,cx=(torch.rand(1,10),torch.rand(1,10))
        hx,cx=self.rnn(input[0],(hx,cx))
        step_output=hx.clone()
        
        length=input.shape[0]
        for i in range(length-1):
            hx,cx=self.rnn(input[i],(hx,cx))
            step_output=torch.cat((step_output, hx), dim=0)
        return step_output

test=mini_attn()
outputs=test(torch.rand(10,1,5))