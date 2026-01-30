import torch

with torch.no_grad():
        model = FRCNNModel()
        model.eval()
        model.cuda()

        print('start to convert...')
        torch.onnx.export(model,
                          x,
                          model_path,
                          verbose=False,
                          opset_version=10,
                          export_params=True)