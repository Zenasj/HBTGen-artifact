3
import traceback

import torch
from transformers import AutoTokenizer, modeling_utils

from minicpmv26 import *
from model_prepare import MiniCPMV25VisualEncoder

hf_model_path = "/data/hf_model/openbmb-MiniCPM-V-2_6"

minicpmv_config = MiniCPMVConfig.from_pretrained(hf_model_path, trust_remote_code=True)
minicpmv_model = MiniCPMV(minicpmv_config)
model_weights = modeling_utils.load_state_dict(f"{hf_model_path}/model-00004-of-00004.safetensors")
minicpmv_model.load_state_dict(model_weights, strict=False)

minicpmv_tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
minicpmv_img_processor = MiniCPMVImageProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
minicpmv_processor = MiniCPMVProcessor(image_processor=minicpmv_img_processor, tokenizer=minicpmv_tokenizer)



dummy_input = torch.randn([3, 3, 14, 14448], dtype=torch.float32)

ve_model = MiniCPMV25VisualEncoder(minicpmv_model.vpm, minicpmv_model.resampler)
ve_model.eval()
ve_model.requires_grad_(False)
ve_model.prepare_resampler()
# ====================算子替换========================
linear_op_out = ve_model(dummy_input)
ve_model.replace_linear_with_conv2d()
ve_model.eval()
ve_model.requires_grad_(False)
conv_op_out = ve_model(dummy_input)
# ====================算子替换========================
del minicpmv_model
try:
    torch.onnx.export(ve_model, dummy_input, "v26dynamo.onnx", dynamo=True, report=True)
except Exception as e:
    traceback.print_stack(e)
    traceback.print_exception(e)

3
traced_graph = torch.export.export(ve_model, args=(dummy_input, ), strict=False)      #
torch.onnx.export(traced_graph, dummy_input, "v26dynamo.onnx", dynamo=True, report=True)

3
def _create_node(
    graph_or_block: _C.Graph | _C.Block,
    domain_op: str,
    inputs: Sequence,
    attributes: dict,
    params_dict: dict,
    opset_version: int,
    n_outputs: int,
    shape_inference: bool = True,
) -> _C.Node:
    """Creates an node 'domain_op', taking inputs and attributes."""
    # ===============origin code from pytorch==================
    if shape_inference:
        print("===============")
        print(node)
        print(params_dict)
        print(opset_version)
        _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
    return node

3
key_padding_mask = torch.zeros(xxx, dtype=torch.bool)
key_padding_mask[0, 1032:] = True