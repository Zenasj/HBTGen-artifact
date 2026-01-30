import torch
import torch.nn as nn
import onnxruntime

device = 'cuda'
use_mask = True
model_path = 'simple_model.onnx' 


### Mask calculation

def compute_output_lengths(x, lengths_fraction=None):
    if lengths_fraction is None:
        return torch.full(x.shape[:1], x.shape[-1], device=x.device, dtype=torch.long)
    return (lengths_fraction * x.shape[-1]).ceil().long()


def temporal_mask(x, lengths):
    return (torch.arange(x.shape[-1], device=x.device, dtype=lengths.dtype).unsqueeze(0) <
            lengths.unsqueeze(1)).view(x.shape[:1] + (1,) * (len(x.shape) - 2) + x.shape[-1:])


### Simple model for export

class SimpleNetwork(nn.Module):
    def __init__(self, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=3,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1
                               )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=3,
                               out_channels=1,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1)

    def forward(self, x, xlen):
        # x - [B; T], xlen - [B]
        x = x.unsqueeze(1)
        # x - [B; 1; T]
        if self.use_mask:
            mask = temporal_mask(x, compute_output_lengths(x, xlen))
            x = x * mask    
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


### Random tensor to export

onnx_sample_batch_size = 16
onnx_sample_time = 1024

waveform_input = torch.rand(onnx_sample_batch_size, onnx_sample_time, device=device)
xlen = torch.rand(onnx_sample_batch_size, device=device)

### Create model

model = SimpleNetwork(use_mask=use_mask).to(device)
result_torch = model(waveform_input, xlen)

### Export model

torch.onnx.export(
    model, (waveform_input, xlen,),
    model_path,
    verbose=False,
    opset_version=12,
    export_params=True,
    do_constant_folding=True,
    input_names=['x', 'xlen'],
    output_names=['logits'],
    dynamic_axes=dict(x={
        0: 'B', 1: 'T'
    }, logits={
        0: 'B', 2: 't'
    }, xlen={
        0: 'B'
    })
)

session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
onnx_input = dict(x=waveform_input.cpu().numpy())
if use_mask:
    onnx_input['xlen'] = xlen.cpu().numpy()
result_onnx = session.run(None, onnx_input)[0]
result_onnx = torch.as_tensor(result_onnx, device=device)

### Сorrectness check

assert torch.allclose(result_torch.cpu(), result_onnx.cpu(), rtol=1e-02, atol=1e-03)

### Doing the same but with different shape

validate_batch_size = 32
validate_sample_time = 512

validate_waveform_input = torch.rand(validate_batch_size, validate_sample_time, device=device)
validate_xlen = torch.rand(validate_batch_size, device=device)

validate_result_torch = model(validate_waveform_input, validate_xlen)

validate_onnx_input = dict(x=validate_waveform_input.cpu().numpy())
if use_mask:
    validate_onnx_input['xlen'] = validate_xlen.cpu().numpy()
validate_result_onnx = session.run(None, validate_onnx_input)[0]
validate_result_onnx = torch.as_tensor(validate_result_onnx, device=device)

assert torch.allclose(validate_result_torch.cpu(), validate_result_onnx.cpu(), rtol=1e-02, atol=1e-03)

def forward(self, x, xlen):
        # x - [B; T]
        # x.squeeze(1) - [B; T]
        # x.squeeze(1).unsqueeze(1) - [B; 1; T]
        x = x.squeeze(1).unsqueeze(1)
        if self.use_mask:
            mask = temporal_mask(x, compute_output_lengths(x, xlen))
            x = x * mask
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x