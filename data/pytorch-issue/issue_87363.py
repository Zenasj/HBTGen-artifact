import torch
def main():
    # initialize
    tensor = torch.randn((3, 3), device='cuda:0', dtype=torch.float32)
    
    # print
    print("Initial Tensor (on GPU 0):")
    print(tensor)
    print("Initial Tensor dtype:", tensor.dtype)
    print("Initial Tensor shape:", tensor.shape)
    print("Initial Tensor device:", tensor.device)
    
    # move tensor from gpu0 to gpu1
    tensor_moved = tensor.to('cuda:1')
    
    # print
    print("\nTensor after moving to GPU 1:")
    print(tensor_moved)
    print("Moved Tensor dtype:", tensor_moved.dtype)
    print("Moved Tensor shape:", tensor_moved.shape)
    print("Moved Tensor device:", tensor_moved.device)

if __name__ == "__main__":
    main()