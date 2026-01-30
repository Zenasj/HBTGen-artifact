import torch
import tensorrt

graph = torch._C.Graph()
graph.addInput()

for i in graph.inputs():
    print(i)

print('finish')