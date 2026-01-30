import torch

def test(pred_scores, score_thresh):
    keep_idxs = pred_scores > score_thresh
    topk_idxs = torch.nonzero(keep_idxs)  # Kx2
    return topk_idxs

if __name__ == "__main__":
    x = torch.tensor(
        [
            [0.1, 0.2002, 0.3, 0.4],
        ]
    ).to(torch.bfloat16)

    # x = torch.tensor(
    #     [
    #         [0.1, 0.2002, 0.3, 0.4],
    #         [0.1, 0.6, 0.7, 0.8],
    #         [0.9, 1.0, 1.1, 1.2],
    #     ]
    # )

    score_thresh = 0.2
    eager_res = test(x, score_thresh)
    cnf = torch.compile(test)
    res = cnf(x, score_thresh)
    # print(torch.allclose(eager_res, res), flush=True)
    print("bf16 eager_res is: {}".format(eager_res), flush=True)
    print("bf16 inductor res is: {}".format(res), flush=True)

cpp_fused_gt_0 = async_compile.cpp_pybinding(['const bfloat16*', 'bool*'], '''
#include "/tmp/torchinductor_leslie/sk/cskh5dx62fglpphcrl6723dnmowdabouerrzy3dmqcngbxwfa7bv.h"
extern "C" void kernel(const bfloat16* in_ptr0,
                       bool* out_ptr0)
{
    {
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.2);
            auto tmp3 = tmp1 > tmp2;
            out_ptr0[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')