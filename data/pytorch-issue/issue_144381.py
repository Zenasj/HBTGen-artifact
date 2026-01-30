import torch
import torch.nn as nn
import numpy as np

python
class DataCov(nn.Module):
    def __init__(self):
        super(DataCov, self).__init__()

        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=1536, hop_length=768, f_min=20, f_max=20000)
        )

    def forward(self, x1):
        return self.transform(x1)

def export_datacov_onnx(path):
    model = DataCov()
    model.eval()
    src_wav = torch.randn((1, 1, 48000 * 12), requires_grad=True)
    input_names = ["wav_data"]
    output_names = ["ans"]
    args = (src_wav,)
    torch.onnx.export(
        model,
        args,
        path,
        export_params=True,
        opset_version=19,
        do_constant_folding=True, 
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        report=True
    )
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

def test_data_cov_onnx(onnx_path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        'CUDAExecutionProvider',
        'DmlExecutionProvider',
        'CPUExecutionProvider'
    ]
    session = ort.InferenceSession(onnx_path, sess_options,
                                   providers=providers)
    src_wav = torch.randn((1, 1, 48000 * 12))
    ort_inputs = {session.get_inputs()[0].name: src_wav.numpy(), }
    ort_outs = session.run(None, ort_inputs)
    ort_outs = ort_outs[0]
    ort_outs = torch.from_numpy(ort_outs)

    model = DataCov()
    model.eval()
    deal_1 = model(src_wav)

    print(f'Torch Output Shape: {deal_1.shape}, ONNX Output Shape: {ort_outs.shape}')
    print(f'Torch Output Min/Max: {torch.min(deal_1)}, {torch.max(deal_1)}')
    print(f'ONNX Output Min/Max: {torch.min(ort_outs)}, {torch.max(ort_outs)}')
    print(f'Torch Output Mean/Std: {torch.mean(deal_1)}, {torch.std(deal_1)}')
    print(f'ONNX Output Mean/Std: {torch.mean(ort_outs)}, {torch.std(ort_outs)}')

    np.testing.assert_allclose(deal_1.detach().numpy(), ort_outs.detach().numpy(), rtol=1e-02, atol=1e-04)

if __name__ == '__main__':
    export_datacov_onnx("DataCov.onnx")
    test_data_cov_onnx("DataCov.onnx")