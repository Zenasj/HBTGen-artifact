import torch
import torchaudio

input_dim = 80
conformer = torchaudio.models.Conformer(
     input_dim=input_dim,
     num_heads=4,
     ffn_dim=128,
     num_layers=4,
     depthwise_conv_kernel_size=31,
)

torch.save(conformer.eval(), "conformer.pt")

lengths = torch.Tensor([10, 20, 12])
input = torch.ones(3, 20, 80)

torch.onnx.export(conformer,
                  f="../conformer.onnx",
                  input_names=["input", "lengths"],
                  output_names=["logits", "output_lengths"],
                  args=({ "input": input, "lengths": lengths }),
                  dynamic_axes={
                    "input": { 0: "batch_count", 1: "batch_item_length" },
                    "lengths": { 0: "batch_count" },
                    "logits": { 0: "batch_count", 1: "batch_item_length" },
                    "output_lengths": { 0: "batch_count" }
                  }
                )

conformer = torch.load("conformer.pt")

lengths = torch.Tensor([10, 20, 12])
input = torch.ones(3, 20, 80)

output = conformer(input, lengths)

import onnxruntime as ort
import numpy

session = ort.InferenceSession("../conformer.onnx")

ortvalue = ort.OrtValue.ortvalue_from_numpy(input.numpy())
ortvalue2 = ort.OrtValue.ortvalue_from_numpy(lengths.numpy())

input_name = session.get_inputs()[0].name
input2_name = session.get_inputs()[1].name
label_name = session.get_outputs()[0].name
label2_name = session.get_outputs()[1].name

result = session.run([label_name, label2_name], { input_name: ortvalue, input2_name: ortvalue2 })

lengths = torch.Tensor([10, 20, 12, 15])
input = torch.ones(4, 20, 80)

lengths = torch.Tensor([10, 25, 12])
input = torch.ones(3, 25, 80)