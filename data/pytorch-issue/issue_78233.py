import torch

input_semantics = torch.randn(1,18,256,256)
degraded_image = torch.randn(1,3,256,256)
if len(self.opt.gpu_ids) > 0:
    input_semantics = input_semantics.cuda(self.opt.gpu_ids[0])
    degraded_image = degraded_image.cuda(self.opt.gpu_ids[0])
with torch.no_grad():
    fake_image = self.netG(input_semantics, degraded_image, z=None)
    torch.onnx.export(self.netG,(input_semantics, degraded_image), 
                  "./Face_Enhancement_Setting_9_epoch_100_latest_net_G_model.onnx",
                  verbose=False,
                  opset_version=13,
                  export_params=True, do_constant_folding=True,
                  input_names=["image"],
                  output_names=["output"]
                  )