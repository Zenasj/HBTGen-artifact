import torch.nn as nn

import torch, time
import torch.cuda.comm
import torch.nn.functional as F

a = torch.randn(4, 8192, 4096)
b = a.clone()
c = a.clone()
x = torch.randn(4, 512, 8192)

tp_num = 4
tp_streams = [torch.cuda.Stream(device=i) for i in range(tp_num)]

a_chunks = torch.cuda.comm.scatter(a, devices=range(tp_num), dim=0)
b_chunks = torch.cuda.comm.scatter(b, devices=range(tp_num), dim=0)
c_chunks = torch.cuda.comm.scatter(c, devices=range(tp_num), dim=0)
x_chunks = torch.cuda.comm.broadcast(x, [i for i in range(tp_num)])

a_results = []
b_results = []
c_results = []

torch.cuda.synchronize()
st_time = time.time()
for i in range(tp_num):
    with torch.cuda.stream(tp_streams[i]):
        a_chunk = a_chunks[i].permute(0, 2, 1)
        a_chunk = torch.flatten(a_chunk, start_dim=0, end_dim=1)
        a_out   = F.linear(x_chunks[i], a_chunk) 

        b_chunk = b_chunks[i].permute(0, 2, 1)
        b_chunk = torch.flatten(b_chunk, start_dim=0, end_dim=1)
        b_out   = F.linear(x_chunks[i], b_chunk) 
    
        c_chunk = c_chunks[i].permute(0, 2, 1)
        c_chunk = torch.flatten(c_chunk, start_dim=0, end_dim=1)
        c_out   = F.linear(x_chunks[i], c_chunk) 
    
        a_results.append(a_out)
        b_results.append(b_out)
        c_results.append(c_out)
        #tp_streams[i].synchronize() #does not affect time
    #torch.cuda.synchronize(i)#Whether to synchronize in series, does not affect time
    
torch.cuda.synchronize()
a_res = torch.cuda.comm.gather(a_results, dim=0, destination='cuda:0')
b_res = torch.cuda.comm.gather(b_results, dim=0, destination='cuda:0')
c_res = torch.cuda.comm.gather(c_results, dim=0, destination='cuda:0')
print(time.time() - st_time)

is_equal = torch.equal(a_res, b_res) and torch.equal(a_res, c_res)
print(is_equal)