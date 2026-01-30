import torch
import torchbenchmark
import torch.fx
import warnings
import time

if __name__ == '__main__':
    device = 'cpu'

    succeeded_count = 0
    succeeded_names = []

    for model_idx, Model in enumerate(torchbenchmark.list_models()):
        if 'nvidia_deeprecommender' not in Model.__module__:
            continue

        try:
            m = Model(device=device)
        except Exception as e:
            warnings.warn(f'{Model.__module__}.{Model.__name__} failed instantiation {str(e)}')
            continue

        float_model, example = m.get_module()

        batch_sizes = [1, 16, 64, 128, 256]
        dataset = [(example[0][:batch_size].contiguous(), None) for batch_size in batch_sizes]

        from torch.quantization import get_default_qconfig
        from torch.quantization.quantize_fx import prepare_fx, convert_fx
        float_model.eval()
        qconfig = get_default_qconfig("fbgemm")
        qconfig_dict = {"": qconfig}
        def calibrate(model, data_loader):
            model.eval()
            with torch.no_grad():
                for image, target in data_loader:
                    model(image)
        s = time.time()
        traced = torch.fx.symbolic_trace(float_model)
        print(traced.code)
        prepared_model = prepare_fx(float_model, qconfig_dict)
        print('prepare', (time.time() - s) * 1000, 'ms')
        s = time.time()
        calibrate(prepared_model, dataset)
        print('calibrate', (time.time() - s) * 1000, 'ms')
        s = time.time()
        quantized_model = convert_fx(prepared_model)
        print('convert', (time.time() - s) * 1000, 'ms')