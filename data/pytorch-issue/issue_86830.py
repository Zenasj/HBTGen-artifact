import torch.distributed as dist
import torch

@torch.inference_mode()
def test_inference():
    tensor_list = [torch.zeros(1) for _ in range(dist.get_world_size())]

    tensor = torch.tensor([dist.get_rank()], dtype=torch.float32)
    dist.all_gather(tensor_list, tensor)
    print(tensor_list)
    
def main():
    dist.init_process_group("gloo")
    print('rank', dist.get_rank())
    test_inference()

if __name__ == '__main__':
    main()