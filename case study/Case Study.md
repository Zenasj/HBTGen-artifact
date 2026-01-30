# Case Study

## [Issue 163064](https://github.com/pytorch/pytorch/issues/163064)
```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# optional: pin to a single GPU for clarity
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 4)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # CPU input
    return torch.randn(1, 10, requires_grad=True)

def repro(case="A"):
    model = my_model_function()

    input_tensor = GetInput()         # CPU tensor
    _ = model(input_tensor)           # eager CPU run OK

    # Compile on CPU
    compiled_cpu = torch.compile(model)

    if case == "B":
        # Calling the CPU-compiled function once seems to change the behavior later
        _ = compiled_cpu(input_tensor)  # OK on CPU

    # Move the SAME module instance to CUDA, and compile again
    model.to("cuda")
    print("before second compile")
    compiled_cuda = torch.compile(model)
    print("after second compile")

    # Now call CUDA-compiled function with a CPU input on purpose
    # Expectation: a clear "device mismatch" style error
    # Actual: either FakeTensor device propagation error (case A),
    #         or a segmentation fault (case B).
    out = compiled_cuda(input_tensor)

if __name__ == "__main__":
    # Case A: Do NOT call the first compiled function before moving to CUDA
    # -> raises: TorchRuntimeError: Unhandled FakeTensor Device Propagation ...
    try:
        repro("A")
    except Exception as e:
        print("Case A raised:", repr(e))

    # Case B: Call the CPU-compiled function once before moving to CUDA
    # -> causes a Segmentation fault (core dumped) on my setup
    # Comment this in/out to observe the difference:
    repro("B")
```

This program exposes a state-dependent cross-device recompilation anomaly in `torch.compile`. The model is compiled on CPU before being moved to CUDA and recompiled again. A single invocation of the CPU-compiled function affects the subsequent CUDA compilation,  causing the same program to either raise a FakeTensor device propagation error or crash with a segmentation fault.

The bug is subtle: device movement and recompilation are semantically independent operations, yet the runtime exhibits order-sensitive failure modes, making the defect difficult for traditional compiler fuzzers or differential testing frameworks to trigger.

## [Issue 163640](https://github.com/pytorch/pytorch/issues/163640)
```python
import torch
import torch.nn as nn

class TinyEnc(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, 10)

    def forward(self, x, pad_mask):
        # Passing src_key_padding_mask triggers torch._nested_tensor_from_mask_left_aligned
        y = self.enc(x, mask=None, src_key_padding_mask=pad_mask)
        return self.proj(y)

def main():
    torch.manual_seed(0)
    m = TinyEnc().eval()

    B, T, C = 1, 41, 512
    x = torch.randn(B, T, C, dtype=torch.float32)
    pad_mask = (torch.rand(B, T) > 0.5)
    pad_mask[..., 0] = True

    # Eager is fine
    with torch.inference_mode():
        y = m(x, pad_mask)
    print("eager ok:", tuple(y.shape))

    # Compile (fullgraph=True required to reproduce)
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    _ = cm(x, pad_mask)

if __name__ == "__main__":
    main()
```
This example shows a compilation failure induced by boolean padding masks in the `nn.allowbreak TransformerEncoder` module under `torch.compile` with full-graph mode enabled.

While eager execution succeeds, full-graph compilation raises an `Unsupported` exception due to an internal call to `_nested_tensor_from_mask_left_aligned` returning a boolean value rather than a tensor.

The failure is systematic and input-independent, yet difficult for traditional test generators to uncover because it requires a specific combination of module structure, invocation pattern, and boolean mask semantics to reach the unsupported path during graph construction.

## [Issue 172213](https://github.com/pytorch/pytorch/issues/172213) 
```python
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_size=10, output_size=20):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.multiplier = nn.Parameter(torch.tensor(2.0))

        # Buffers
        self.register_buffer("flag", torch.tensor(True, dtype=torch.bool))
        self.register_buffer("step", torch.tensor(0, dtype=torch.int64))

    def forward(self, x):
        # In-place update of a registered buffer
        self.step += 1

        x = x + 1
        x = self.linear(x)
        x = x * self.multiplier

        if self.flag.bool():
            x = x + torch.ones_like(x)
        return x


def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    model = MyModel()
    input_data = torch.randn(1, 10)

    # Sanity check: eager forward works
    _ = model(input_data)

    # Compile the model (default backend, default options)
    compiled_model = torch.compile(model)

    # Forward + backward through compiled model
    output_compiled = compiled_model(input_data)
    loss = output_compiled.sum()
    loss.backward()

    # Extra compiled forwards (kept to match the reproducing script)
    for _ in range(3):
        compiled_model(input_data)

    # Try export; failure is expected and ignored
    try:
        torch.export.export(model, (input_data,))
    except Exception:
        pass

    # Eager forward *after* compile + backward + export:
    # this triggers:
    #   AssertionError: assert isinstance(buffer, FakeTensor)
    # in torch/_functorch/_aot_autograd/utils.py::_map_assigned_buffer_to_proxy
    model.eval()
    model(input_data)


if __name__ == "__main__":
    main()
```
This program reveals a cross-mode inconsistency involving buffer mutation under `torch.allowbreak compile`.
Eager execution and compilation both succeed, and backward propagation through the compiled graph also completes without error. However, a subsequent eager invocation of the same module raises an `AssertionError` at runtime `assert isinstance(buffer, FakeTensor)` originating from the AOTAutograd buffer mapping logic.

The failure manifests only after a specific sequence of operations—compile, forward, backward, and re-enter eager mode—with no dependence on the concrete input tensor, making it challenging for conventional test generators to discover from interface signatures alone.

## [Issue 171673](https://github.com/pytorch/pytorch/issues/171673)

```python
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.win = torch.hann_window(320)
        self.deconv = nn.ConvTranspose2d(2, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = torch.stft(
            x,
            n_fft=512,
            hop_length=160,
            win_length=320,
            window=self.win,
            return_complex=True,
            pad_mode="constant",
        )
        y = torch.view_as_real(y)          # (B, F, T, 2)
        y = y.permute(0, 3, 1, 2)          # (B, 2, F, T)
        y = self.deconv(y)                 # (B, 3, F, T)
        return y

def main():
    m = M().eval()
    x = torch.randn(1, 1024)

    ep = torch.export.export(
        m,
        (x,),
        dynamic_shapes=({0: torch.export.Dim("batch"), 1: torch.export.Dim("time")},),
        strict=False,
    )

    y1 = m(x)
    y2 = ep.module()(x)
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
    print("OK")

if __name__ == "__main__":
    main()
```

This program exercises a model that applies a short-time Fourier transform to a dynamically shaped input, followed by a real/imaginary view conversion, a permutation, and a `ConvTranspose2d` operation.

Eager execution succeeds, but exporting the same model with symbolic dynamic shapes triggers an AssertionError from the symbolic shape solver (`assert op == "=="`) during `symbolic_shapes.prettify_results`, indicating an internal invariant violation.

The crash is not input-dependent and occurs before replaying the exported module, arising only under dynamic shape export on the STFT pipeline.
The failure requires a specific combination of operator sequence and export configuration rather than particular argument values, making it difficult for conventional test generators to reproduce from interface signatures alone.