ERROR: test_DistributedDataParallel (__main__.TestDistBackendWithFork)
ERROR: test_DistributedDataParallel_SyncBatchNorm (__main__.TestDistBackendWithFork)
ERROR: test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_gradient (__main__.TestDistBackendWithFork)
ERROR: test_DistributedDataParallel_with_grad_is_view (__main__.TestDistBackendWithFork)

PTSingle: 0.6859778165817261
PT    :   0.6859777371088663
NPSing:   0.6859777629530678
NP    :   0.6859777629530678

import torch
import numpy as np

data = [0.0014468701556324959,0.19974617660045624,0.2590259611606598,3.7469475269317627,0.027207523584365845,0.2559516727924347,0.1512461155653,2.632812023162842,0.1647290289402008,0.4095090329647064,0.24858342111110687,0.1345278024673462]
ptTensorCPU = torch.Tensor(data)
ptTensorGPU = ptTensorCPU.cuda()

print('np : %s' % np.mean(data))
print('cpu: %s' % float(ptTensorCPU.mean()))
print('gpu: %s' % float(ptTensorGPU.mean()))