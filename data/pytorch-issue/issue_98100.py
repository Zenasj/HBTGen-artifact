import torch

torch.manual_seed(420)

x = torch.randn(3, 3)
y = torch.randn(3, 3)

d = {"x":x, "y":y}
model = CustomAdd()
model.eval()

pt_outputs = model(d)
print('From PyTorch:', pt_outputs)

input_names = ["x", "y"]
torch.onnx.export(model, (d, {}), "add.onnx", input_names=input_names)

ort_sess = onnxruntime.InferenceSession("add.onnx")
ort_inputs = {"x":x.cpu().numpy(), "y":y.cpu().numpy()}
ort_outputs = ort_sess.run(None, ort_inputs)
print('From onnxruntime:', ort_outputs)

for input in ort_sess.get_inputs():
    print(input.name)

ort_inputs = {ort_sess.get_inputs()[0].name: x, ort_sess.get_inputs()[1].name: y}