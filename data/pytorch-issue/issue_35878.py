import torch

def package():
    model = BiSINet(p=2, q=8)
    model.load_state_dict(
        weight_convert(torch.load("weights/temp/SINetDIS_decoder_360_3c.pth", "cpu")))
    model.cpu()
    model.eval()
    with torch.no_grad():
        model_input = torch.rand(1, 3, 224, 224)
        trace_model_script = torch.jit.trace(model, model_input)
        trace_model_script.save('weights/convert/SINet_decoder_3c_x86.pt')