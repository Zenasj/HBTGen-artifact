import torch
import math

for i in range(2):
    m, n = math.floor(torch.iinfo(torch.int32).max/n)+i, 128
    print("total bigger than int32", (torch.iinfo(torch.int32).max/n<m))
    orig_arr = torch.rand((m, n), dtype=torch.float32)
    observer_symmetric = torch.quantization.MinMaxObserver(
        qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
    )

    observer_symmetric(orig_arr)
    scale, zero = observer_symmetric.calculate_qparams()

    qat_arr = torch.quantize_per_tensor(
        orig_arr, scale=scale, zero_point=zero, dtype=torch.qint8
    )

    print("quantized",qat_arr[0:5,0:5])
    print("normal",orig_arr[0:5,0:5])