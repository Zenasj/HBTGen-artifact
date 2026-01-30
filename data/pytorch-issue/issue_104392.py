import torch

state_dict = load_state_dict(torch.load(paths))
model_cls = TestModel()
model = model_cls.from_state_dict(state_dict).eval()
opset_version = 18
onnx_file_name = paths.split('.')[0]
dummy_input = torch.rand(1, 3, 512, 768) 
torch.onnx.export(model, 
                  dummy_input, 
                   #onnx_file_name+".onnx",
                  opset_version=opset_version)

opset_version = 18
custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=1)
@onnxscript.script(custom_opset)
def erfc(x):
    y = 1-op.erf(x)
    return y  

def custom_erfc(g: jit_utils.GraphContext, x):
    return g.onnxscript_op(erfc).setType(x.type())

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::erfc",
    symbolic_fn=custom_erfc,
    opset_version=opset_version,
)

onnx_model = onnx.load("model.onnx")
onnx_model_graph = onnx_model.graph
onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

input_shape = (1, 3, 500, 500)
x = torch.randn(input_shape).numpy()
onnx_output = onnx_session.run("output")