import torch
import torch.onnx

feat = torch.randn((2708, 1433), requires_grad=True)
adj = torch.randn((2708, 2708), requires_grad=False)
model = torch.load("saved_KipfGCN_model.pt")

def make_ONNX():
    torch.onnx.export(
        model, 
        (feat, adj), 
        'model.onnx',
        export_params=True, 
        opset_version=10,
        do_constant_folding=True, 
        input_names=['features', 'adjacency'], 
        output_names=['classify_7'],
        enable_onnx_checker=True,
        _retain_param_name=True
    )

def main():
    print(feat.shape)
    print(adj.shape)
    make_ONNX()

if __name__ == '__main__':
    main()