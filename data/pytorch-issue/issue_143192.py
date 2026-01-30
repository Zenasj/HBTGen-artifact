import torch
import torch.nn as nn

class TwoLayerNetDynamic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNetDynamic, self).__init__()
        self.model_name = 'TwoLayerNetDynamic'

        self.fully_connected_1 = nn.Linear(input_size, hidden_size)
        self.early_exit_head_1 = nn.Linear(hidden_size, output_size)

        self.threshold = torch.tensor([0.0], dtype=torch.float32).to(DEVICE)
        self.last_exit = nn.Linear(hidden_size, output_size)

        self.training_exits = False

        print("TwoLayerNetDynamic initialized")

    def forward(self, x: torch.Tensor):
        mean = x.mean()
        
        # version 1 
        # if torch.gt(mean, self.threshold):
        #     x = self.early_exit_head_1(x)
        # else:
        #     x = self.last_exit(x)
        # x = torch.cat([x, mean.reshape_as(x)], dim=1)

        # version 2
        x = self.fully_connected_1(x)
        x = fc.cond(mean>0.0,self.early_exit_head_1,self.last_exit,(x,))
        x = torch.cat([x, mean.reshape_as(x)], dim=1)       
        return x

### Using TorchDynamo ###
filename = f""
onnx_filepath = f"./models/onnx/{model.model_name}_dynamo.onnx"
onnx_program:ONNXProgram = torch.onnx.export(
  model=model,
  args=(_x,),
  dynamo=True,
  report=True
)
onnx_program.save(onnx_filepath)