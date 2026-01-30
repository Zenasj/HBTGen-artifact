import torch
import torch.nn as nn
import numpy as np

class upsampleTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.up1 = simplenet.UpsampleBlock(32, 32, 64)
        self.up2 = simplenet.UpsampleBlock(64, 32, 64)


    def forward(self, inp, skip, skip2):
        up = self.up1(inp, skip)
        up2 = self.up2(up, skip2)
        return up2


def test_upsample_block2():
    skip2 = torch.rand(1,32,640,368)
    skip = torch.rand(1,32, 320, 184)
    inp = torch.rand(1,32, 160, 92)

    model_name = "upsample_block2"
    blk2 = upsampleTest()
    m = blk2
    m.eval()
    pt_out = m(inp, skip,skip2)
    with torch.no_grad():
        trace_model = torch.jit.trace(m, (inp,skip,skip2))
    trace_model.save(f"{model_name}.pt")

    
    ts_out = trace_model(inp, skip,skip2)

    torch.onnx.export(trace_model.cpu(), (inp.to("cpu"), skip.to("cpu"),skip2.to("cpu")), f"{model_name}.onnx", input_names=["in", "skip","skip2"], 
                    keep_initializers_as_inputs=False,
                    do_constant_folding=True,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                    opset_version=12,
                    verbose=True
    )

    ort_session = ort.InferenceSession(f"{model_name}.onnx", providers=['CUDAExecutionProvider'])

    onnx_out = ort_session.run(
        None,
        {"in": inp.cpu().numpy().astype(np.float32),
         "skip": skip.cpu().numpy().astype(np.float32),
         "skip2": skip2.cpu().numpy().astype(np.float32)},
    )

    print(f"{model_name}")
    print("onnx", np.mean(onnx_out[0]))
    print("ts", np.mean(ts_out[0].detach().cpu().numpy()))
    print("pt", np.mean(pt_out[0].detach().cpu().numpy()))
    print("diff", np.mean(onnx_out[0]-pt_out[0].detach().cpu().numpy()))
    print("ts onnx diff", np.mean(onnx_out[0]-ts_out[0].detach().cpu().numpy()))

test_upsample_block2()

class UpsampleBlock(nn.Module):
  def __init__(self, in_channel, skip_in_channel, out_channel):
    super().__init__()
    
    self.layers = nn.Sequential(
      ConvBlock(in_channel, out_channel, k=1)
    )

    self.skip_layers = nn.Sequential(
      ConvBlock(skip_in_channel, out_channel, k=1)
    )

    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
  
  def forward(self, inp, skip):
    # pdb.set_trace()
    inp = self.layers(inp)
    skip = self.skip_layers(skip)
    inp = self.up(inp)
    return inp + skip