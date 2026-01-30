import torch.nn as nn

import torch as th
import onnx
import onnxruntime


model = th.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
dummy_input = th.randn(1, 3, 256, 256, dtype=th.float32)
output = model(dummy_input)


# We trace the model and convert it right away : JIT model check is OK
traced_model = th.jit.trace(model, dummy_input)
th.onnx.export(traced_model, dummy_input, "ts_nosave_upsample.onnx",
               example_outputs=(output), verbose=True, opset_version=12)
onnx_model = onnx.load("ts_nosave_upsample.onnx")
onnx.checker.check_model(onnx_model, full_check=True)
ort_session = onnxruntime.InferenceSession("ts_nosave_upsample.onnx")


# Here we save the modules first, load them..
th.save(model, "upsample.pt")
th.jit.save(traced_model, "upsample.ts")
model = th.load("upsample.pt")
traced_model = th.jit.load("upsample.ts")

# and only then convert to ONNX..
th.onnx.export(model, dummy_input, "pt_upsample.onnx",
               example_outputs=(output), verbose=True, opset_version=12)
th.onnx.export(traced_model, dummy_input, "ts_upsample.onnx",
               example_outputs=(output), verbose=True, opset_version=12)

# so we can load them afterwards
pt_onnx_model = onnx.load("pt_upsample.onnx")
ts_onnx_model = onnx.load("ts_upsample.onnx")

# This model is OK
onnx.checker.check_model(pt_onnx_model, full_check=True)
ort_session = onnxruntime.InferenceSession("pt_upsample.onnx")

# Fails : ----> type inconsistency error in mul nodes (float64 instead of float32) <----
onnx.checker.check_model(ts_onnx_model, full_check=True)
ort_session = onnxruntime.InferenceSession("ts_upsample.onnx")