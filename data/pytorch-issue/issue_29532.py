from __future__ import absolute_import, division, print_function, unicode_literals

import time

import torch

K, N = 1024, 1024

print("M, nthread=1, nthread=2, nthread=4, nthread=8, nthread=16")

for M in (2, 20, 200, 500, 1024,):
    print(M, sep=",", end=", ")
    for num_threads in (1, 2, 4, 8, 16):

        torch.set_num_threads(num_threads)

        x = torch.rand(M, K)
        w = torch.rand(K, N)
        b = torch.rand(N)

        NITER = 20

        W_int8, col_offsets, W_scale, W_zp = torch.fbgemm_linear_quantize_weight(w)
        W_prepack = torch.fbgemm_pack_quantized_matrix(W_int8, W_int8.size(1), W_int8.size(0))

        s = time.time()
        for _ in range(NITER):
            Y_fp32 = torch.fbgemm_linear_int8_weight(x, w, W_prepack, col_offsets, W_scale, W_zp, b)
        elapsed_per_iter_dyn_quant = (time.time() - s) / NITER

        print(
            "{:0.2f}".format(2.0 * M * N * K / elapsed_per_iter_dyn_quant / 1e9),
            end=", ",
        )
    print("\n", end="")