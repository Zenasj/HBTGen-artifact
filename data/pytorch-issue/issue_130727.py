import torch

def main() -> None:
    gpu_device = torch.device("cuda")

    input_tensor = torch.rand(
        (1920, 1, 100),
        device=gpu_device,
        dtype=torch.bfloat16,
    )
    input_tensor = torch.as_strided(
        input_tensor, size=(1920, 1, 100), stride=(100, 100, 1)
    )
    batch1_tensor = torch.rand(
        (1920, 256, 512),
        device=gpu_device,
        dtype=torch.bfloat16,
    )
    batch1_tensor = torch.as_strided(
        batch1_tensor, size=(1920, 256, 512), stride=(512, 983040, 1)
    )
    batch2_tensor = torch.rand(
        (1920, 512, 100),
        device=gpu_device,
        dtype=torch.bfloat16,
    )
    batch2_tensor = torch.as_strided(
        batch2_tensor, size=(1920, 512, 100), stride=(51200, 100, 1)
    )

    for _ in range(100):
        _ = torch.baddbmm(input_tensor, batch1_tensor, batch2_tensor)


if __name__ == "__main__":
    main()

batch1_tensor = torch.as_strided(
        batch1_tensor, size=(1920, 256, 512), stride=(512, 983040, 1)
    )