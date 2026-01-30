import os
import torch
import torch.distributed as dist
import time

def main():
    # 初始化进程组（后端使用NCCL优化GPU通信）
    dist.init_process_group(backend="nccl")

    # 获取当前进程信息
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # 设置当前GPU设备
    torch.cuda.set_device(local_rank)

    # 创建不同进程的初始张量（每个GPU生成不同数据）
    tensor = torch.randn((10000,10000), device=f'cuda:{local_rank}') * (rank + 1)

    #print(f"Rank {rank} 初始数据: {tensor.cpu().numpy()}")

    # 执行all_reduce求和操作（同步所有GPU数据）
    # dist.barrier()
    for i in range(3):
        a = time.time()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        b = time.time()
        #time.sleep(3)
        if work.is_completed():  # 检查工作是否完成
            print(f"rank:{rank}, duration:{work._get_duration()}ms")
        else:
            print("Work did not succeed!")
        #print(f"Rank {rank} 聚合结果: {tensor.cpu().numpy()}")
        print(f"setp: {i}, Rank {rank} AllReduce耗时: {(b - a)*1000:.4f}ms")
        time.sleep(3)
        if local_rank == 0:
            print("-------------------------------------------------------------------------------------")
    # 清理进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

import os
import torch
os.environ['TORCH_NCCL_ENABLE_TIMING'] = "1"
torch.__version__ # '2.4.1+cu121'

for i in range(3):
        a = time.time()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        b = time.time()
        time.sleep(3)
        if work.is_completed() is True:  # 检查工作是否完成
            print(tensor.sum())
            # print(f"step:{i}, rank:{rank}, duration:{work._get_duration()}ms, AllReduce耗时: {(b - a)*1000:.4f}ms")
        else:
            print("Work did not succeed!")
        #print(f"Rank {rank} 聚合结果: {tensor.cpu().numpy()}")
        #print(f"setp: {i}, Rank {rank} AllReduce耗时: {(b - a)*1000:.4f}ms")
        time.sleep(3)
        if local_rank == 0:
            print("-------------------------------------------------------------------------------------")