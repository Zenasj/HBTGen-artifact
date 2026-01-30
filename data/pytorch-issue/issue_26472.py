import torch

@parse_args('v', 'i', 'v', 'v')
def scatter_add(g, self, dim, index, src):
    if self.type().kind() != "CompleteTensorType":
        return _unimplemented("scatter_add", "input size not accessible")
    dtype = self.type().scalarType()
    dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    dims = self.type().sizes()
    to_add = torch.zeros(dims)
    to_add = g.op("Constant", value_t=to_add)
    to_add = scatter(g, to_add, dim, index, src)
    return add(g, self, to_add)

w = torch.scatter(w, 0, idx, w[idx] + input_x[idx])  # w[idx] += input_x[idx]

onnx_model = onnx.load(onnxfile)

graph = onnx_model.graph
nodes = graph.node

# for debugging
count = 0
for node in nodes:
    if not node.name:
        node.name = "__" + node.op_type + "_" + str(count)
        count = count + 1
onnx.save(onnx_model, onnxfile)