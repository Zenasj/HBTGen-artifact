import torch
import onnxruntime
import torch.nn as nn
import numpy as np
import onnx

@torch.jit.script
def fun1(arg1, arg2, arg3:int):
    device = arg1.device
    checkvar = arg1.clone().detach()
    arg2_result = arg2.clone().detach()
    boolvar = torch.zeros(1, dtype=torch.bool, device=device)

    boolvar[0] = False
    for idx in range(int(arg2.item())):
        if (checkvar[idx * 7 + arg3 + 1] == 0):
            boolvar[0] = True
            break
    if boolvar.item() is False:

        checkvar[arg2_result * 5 + arg3] = 356

        arg2_result = arg2_result + 1

    return checkvar, arg2_result

@torch.jit.script
def fun1_batch(var3, arg2, var13:int):

    tvar1 = var3.clone().detach()
    tvar2 = arg2.clone().detach()
    for b_idx in range(tvar1.shape[1]):
        tvar1[:, b_idx], tvar2[b_idx] = fun1(tvar1[:, b_idx], tvar2[b_idx], var13)
    return tvar1, tvar2

class dummyclass(nn.Module):

    def forward(self, input0, input1, input2, input3):

        device = input0.device
        arg2 = input2.clone().detach()
        var3 = input0.clone().detach()
        var3, arg2 = fun1_batch(var3, arg2, 1)

        return input0, input1

print(torch.__version__)

device = torch.device('cpu')

list_var = []

list_var.append(torch.ones( (53,1), dtype=torch.int32, device = device))
list_var.append(torch.ones( (18,1,4), device = device))
list_var.append(torch.ones( (1), dtype=torch.int64, device = device))
list_var.append(torch.ones( (14,1), dtype=torch.int32, device = device))


classvar = dummyclass() 
result = classvar(list_var[0], list_var[1], list_var[2], list_var[3])

torch.onnx.export(classvar, ( list_var[0], list_var[1], list_var[2], list_var[3]), "bugrepro.onnx",
                example_outputs=result,
                opset_version=12,
                verbose=True)


onnx_model = onnx.load("bugrepro.onnx")
problems = onnx.checker.check_model(onnx_model)
print("Exported model has been tested with ONNXRuntime, and the result looks good! As you can see probelms are: " + str(problems))
ort_session = onnxruntime.InferenceSession("bugrepro.onnx")
print("All good here")