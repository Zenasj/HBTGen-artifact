import torch
from torch.onnx import register_custom_op_symbolic

def custom_exp(x):
    """Custom implementation of the exponential function."""
    sign = torch.sign(x)
    abs_x = torch.abs(x)
    exp_int = torch.floor(abs_x)
    exp_frac = abs_x - exp_int
    exp_approx = torch.pow(2.0, exp_int)
    exp_residual = torch.exp(exp_frac * torch.log(torch.tensor(2.0)))
    return sign * exp_approx * exp_residual

def custom_exp_symbolic(symbolic_graph, inputs, attrs, opset_version):
    """Symbolic function for custom_exp."""
    return symbolic_graph.op("CustomExp", inputs)

# Register custom op with PyTorch
register_custom_op_symbolic("aten::exponential", custom_exp_symbolic, 13)

"""main function"""

img = torch.randn(1, 11 ,3, 256, 512).cuda()
imu = torch.randn(1, 101, 6).cuda()


# Export the model
torch.onnx.export( 
	model,               # model being run
	(img, imu), # model input (or a tuple for multiple inputs)
	"tester.onnx",   # where to save the model (can be a file or file-like object)
	export_params=True,        # store the trained parameter weights inside the model file
	opset_version=13,          # the ONNX version to export the model to
	do_constant_folding=True,  # whether to execute constant folding for optimization
	input_names = ['img', 'imu'],   # the model's input names
	output_names = ['output'] # the model's output names
)