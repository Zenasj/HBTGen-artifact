import torch.nn as nn

import onnxscript
from onnxscript.onnx_opset import opset18 as op
# Assuming you use opset18
import torch
import onnx
import torch.onnx
custom_opset = onnxscript.values.Opset(domain="torch_onnx", version=18)


#'LGamma'
@onnxscript.script(custom_opset)
def LGamma(X):
    abs_gamma_x = op.Abs(X)
    output = op.Log(abs_gamma_x)
    return output

def custom_lgamma(g, X):
    return g.onnxscript_op(LGamma, X).setType(X.type())

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::lgamma",  
    symbolic_fn=custom_lgamma,
    opset_version=18,  
)

import torch
import onnx
import torch.onnx
from transformers import AutoTokenizer
import onnx.utils
from onnx import helper, shape_inference

model_path = '/content/drive/MyDrive/CHEF/latent_rationale/chefmodel.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

batch_size = 1
sequence_length = 128
dummy_input_claim = torch.randint(0, 100, (batch_size, sequence_length), dtype=torch.long).to(device)
dummy_input_evidence = torch.randint(0, 100, (batch_size, 5, sequence_length), dtype=torch.long).to(device)

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, claim_input, evidence_input):
        labels = torch.empty(1, dtype=torch.long, device=claim_input.device)
        with torch.no_grad():
            batch = (claim_input, evidence_input, labels)
            logits = self.model(batch)
            predictions = self.model.predict(logits)
        return predictions

wrapped_model = WrappedModel(model)
wrapped_model.train(False)
wrapped_model.eval()

for p in wrapped_model.parameters():
   p.requires_grad_(False)

dummy_input = (dummy_input_claim, dummy_input_evidence)

#tmodule = torch.jit.trace(wrapped_model , dummy_input)
#smodule = torch.jit.script(wrapped_model)
# to onnx
torch.onnx.export(wrapped_model,
         dummy_input,
         'onnx_model.onnx',
         export_params=True,
         verbose=True,
         input_names=["claim_input", "evidence_input"],
         output_names=["output"],
         opset_version=18,
         #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
         #custom_opsets = {"torch.onnx": 18},
                  )

onnx_model = onnx.load('./onnx_model.onnx')  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model