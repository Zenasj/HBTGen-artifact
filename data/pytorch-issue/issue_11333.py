import resource
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import gc
import time

def gpu(item, use_GPU=True):
    if use_GPU:
        return item.cuda()
    else:
        return item

def test_multivariate(loc_mat, cov_mat):
    standard_normal_dist = MultivariateNormal(loc=loc_mat,
                                              covariance_matrix=cov_mat)

def test_potrf(loc_mat, cov_mat):
    n = cov_mat.size(-1)
    [m.potrf(upper=False) for m in cov_mat.reshape(-1, n, n)]

def loop(fn_ref, n=10, size=10000, use_GPU=True, covariance_with_batch_dim=True, print_n=1):
    cov_mat = gpu(torch.zeros((2, 2)) + torch.eye(2), use_GPU)
    if covariance_with_batch_dim:
        cov_mat = cov_mat.unsqueeze(0).repeat(size, 1, 1)
    loc_mat = gpu(torch.zeros((size, 2)), use_GPU)

    for i in range(n):
        gc.collect()
        fn_ref(loc_mat, cov_mat)
        if i % print_n == 0:
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':

    n = 10
    fn_ref = test_multivariate
    use_GPU = True
    covariance_with_batch_dim = True
    print_n = n / 10

    print("test_fn: ", fn_ref.__name__)
    print("n: ", n)
    print("use_GPU: ", use_GPU)
    print("covariance_with_batch_dim: ", covariance_with_batch_dim, "\n")

    start = time.time()

    loop(fn_ref=fn_ref, use_GPU=use_GPU, covariance_with_batch_dim=covariance_with_batch_dim, n=n, print_n=print_n)

    print("\ntime: ", (time.time() - start) / n)

test_fn:  test_multivariate
n:  5
use_GPU:  False
covariance_with_batch_dim:  True 

118808
119200
119552
119908
120268

time:  0.08764233589172363

test_fn:  test_multivariate
n:  5
use_GPU:  True
covariance_with_batch_dim:  True 

1477808
1479680
1479996
1480312
1480684

time:  27.707138347625733

test_fn:  test_multivariate
n:  5000
use_GPU:  False
covariance_with_batch_dim:  False 

107904
107904
109436
109436
109700

time:  0.008288921689987183

test_fn:  test_multivariate
n:  50
use_GPU:  True
covariance_with_batch_dim:  False 

1467364
1468676
1468676
1468680
1468680

time:  0.037353959083557126

test_fn:  test_potrf
n:  5
use_GPU:  False
covariance_with_batch_dim:  True 

116984
117300
117672
118032
118380

time:  0.06480374336242675

test_fn:  test_potrf
n:  5
use_GPU:  True
covariance_with_batch_dim:  True 

1476688
1477008
1477320
1477916
1478168

time:  26.757283926010132

test_fn:  test_potrf
n:  50000
use_GPU:  False
covariance_with_batch_dim:  False 

108420
109720
109984
110248
110512

time:  0.008067794122695923

test_fn:  test_potrf
n:  5
use_GPU:  True
covariance_with_batch_dim:  False 

1468760
1469664
1469680
1469680
1469680

time:  0.2711141109466553