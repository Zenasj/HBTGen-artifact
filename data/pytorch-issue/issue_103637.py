import torch

@torch.compile(fullgraph=True, dynamic=True)
def shift_right(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor >> 2).to(torch.long)

def main():
    sample_input = torch.tensor([4, 4, 16, 32], dtype=torch.uint8)
    print(shift_right(sample_input))

if __name__ == "__main__":
    main()