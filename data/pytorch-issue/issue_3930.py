import time                                                 
import torch                                                
from torch.cuda.comm import broadcast_coalesced             
                                                            
params = [torch.cuda.FloatTensor(3500) for i in range(3000)]
                                                            
for i in range(10):                                         
    start = time.time()                                     
    broadcast_coalesced(params, [0, 1])                     
    torch.cuda.synchronize()                                
    end = time.time()                                       
    print(((end - start) * 1000), 'ms')                     
                                                            
print('-' * 80)                                             
flat_param = torch.cat(params)                              
for i in range(10):                                         
    start = time.time()                                     
    broadcast_coalesced([flat_param], [0, 1])               
    torch.cuda.synchronize()                                
    end = time.time()                                       
    print(((end - start) * 1000), 'ms')