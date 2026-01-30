torch.div(torch.tensor([1,2,3,4],dtype=torch.int64), 400, rounding_mode="floor")
# works as expected: prints tensor([0, 0, 0, 0])
torch.div(torch.tensor([1,2,3,4],dtype=torch.int64).to("mps"), 400, rounding_mode="floor")
# python[41152:6175911] Error getting visible function: 
# (null) Function floorOp_i64 was not found in the library
# /AppleInternal/Library/BuildRoots/20d6c351-ee94-11ec-bcaf-7247572f23b4/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Utility/MPSKernelDAG.mm:755: failed assertion `Error getting visible function: 
 #(null) Function floorOp_i64 was not found in the library' 
# Process finished with exit code 134 (interrupted by signal 6: SIGABRT)

import torch
torch.backends.mps.is_available()
# True