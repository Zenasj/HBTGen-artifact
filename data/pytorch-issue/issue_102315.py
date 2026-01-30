import torch.nn as nn

#handcrafted repro, as no crash happens
import torch
print(torch.__version__) #2.1.0.dev20230525+cu118
class IfModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emptylist = torch.nn.ModuleList()
  def forward(self):
    if self.emptylist:
      print("True")
    else:
      print("False")
    return

non_compiled = IfModule()
compiled = torch.compile(IfModule(), backend="eager")

print("Normal:")
non_compiled() #False
print("Compiled:")
compiled() #True

print("Related:")
print("Non compiled:")
print(bool(torch.nn.ModuleList())) #False
print("Compiled:")
print(bool(torch.compile(torch.nn.ModuleList()))) #True

def x():
  print(True if torch.nn.ModuleList() else False)
torch._dynamo.reset()
torch.compile(x, backend='eager')()#true

def x():
  print(bool(torch.nn.ModuleList()))
torch._dynamo.reset()
torch.compile(x, backend='eager')()#false

print(torch.__version__) #2.1.0.dev20230528+cu118
print("Issue 1: Change of truthiness of directly compiled module")
torch._dynamo.reset()
print("Non compiled:")
print(bool(torch.nn.ModuleList())) #False
print("Compiled:")
print(bool(torch.compile(torch.nn.ModuleList()))) #True

print("Issue 2: Change of if condition resolution")
def x():
  print(True if torch.nn.ModuleList() else False)
torch._dynamo.reset()
print("Non compiled:")
x() #False
print("Compiled:")
torch.compile(x, backend='eager')() #True

print("Issue 3: if(x) non-equivalent with if(bool(x)) - compare with output in Issue 2")
def x():
  print(True if bool(torch.nn.ModuleList()) else False)
torch._dynamo.reset()
print("Non compiled:")
x() #False
print("Compiled:")
torch.compile(x, backend='eager')() #False

UnspecializedNNModuleVariable

UserDefinedVariable