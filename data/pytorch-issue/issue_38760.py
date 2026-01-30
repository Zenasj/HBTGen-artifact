import torch
import encoding

# Get the model
model = encoding.models.get_model('Encnet_ResNet50s_PContext', pretrained=True).cuda()
model.eval()
dummy_input = (
	torch.zeros(
	    32,
	    3,
	    375,
	    500,
	    requires_grad=False,
	)
	.float()
	.cuda()
)
with torch.no_grad():
	torch.onnx.export(
	    model,  # model being run
	    dummy_input,  # model input (or a tuple for multiple inputs)
	    "/veesion/PyTorch-Encoding/segmenter.onnx",
	    input_names=["input"],  # the model's input names
	    output_names=["output"],  # the model's output names
	    export_params=True,
	    verbose=True,
	)