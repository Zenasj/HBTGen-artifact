import torch

B, N, L = 1000, 100, 100000
# this would require 40GiB (and I don't have)
lhs = torch.arange(0, B+N+L, device='cpu', dtype=torch.float32).as_strided([B, N, L], [1,1,1]);
rhs = torch.arange(0, B+L+1, device='cpu', dtype=torch.float32).as_strided([B, L, 1], [1,1,1]);
x = torch.bmm(lhs, rhs)

B, N, L = 1000, 100, 100000
lhs = torch.arange(0, B+N+L, device='cuda', dtype=torch.float32).as_strided([B, N, L], [1,1,1]);
rhs = torch.arange(0, B+L+1, device='cuda', dtype=torch.float32).as_strided([B, L, 1], [1,1,1]);
x = torch.bmm(lhs, rhs) # here it tries to alloocate 37.25GiB

def gemmStridedBatched(handle, 
                      transA, transB,
                      M, N, K, 
                      alpha,
                      A, ldA, strideA, 
                      B, ldB, strideB, 
                      beta,
                      C, ldC, strideC,
                      batchCount):
    for p in range(batchCount):
        for m in range(M):
            for n in range(N):
                c_mnp = sum(A[m + k*ldA + p*strideA] * B[k + n*ldB + p*strideB] 
                           for k in range(K))
                C[m + n*ldC + p*strideC] = alpha*c_mnp + beta*C[m + n*ldC + p*strideC];

def bmm(A, B):
    strideA, ldA, a1 = A.stride()
    strideB, ldB, b1 = B.stride()
    assert(a1 == 1)
    assert(b1 == 1)
    
    bA, mA, nA = A.shape
    bB, mB, nB = B.shape
    assert(bA == bB or bA == 1 or bB == 1) # bash broadcastable
    assert(nA == mB) # matrix multiplication constraint
    
    # fixes the index for 1-element batches
    if bA == 1:
        strideA = 0
    if bB == 1:
        strideB = 0;
    
    batchCount = max(bA, bB)
    
    
    C = torch.empty((batchCount, mA, nB), dtype=torch.float32);
    strideC, ldC, c1 = C.stride()
    assert(c1 == 1)
    
    # view the underlying data as an array
    A_1d = A.as_strided((strideA * bA + ldA * mA + nA,), (1,))
    B_1d = B.as_strided((strideB * bB + ldB * mB + nB,), (1,))
    
    gemmStridedBatched(handle=None, transA=False, transB=False, # not used here
                        M=mA, N=nB, K=mB, alpha=1.0,
                        # input matrices
                        A=A_1d, ldA=ldA, strideA=strideA,
                        B=B_1d, ldB=ldB, strideB=strideB,
                        beta=0.0,
                        # output
                        C=C.view(-1), ldC=ldC, strideC=strideC,  
                        batchCount=batchCount
                     )
    return C