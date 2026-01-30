import torch
import gc


def test_operations(iterations: int, shape: tuple[int, int]) -> None:
    print(f"PyTorch version: {torch.__version__}")
    # Test 1: torch.unique
    print("\nTest 1: torch.unique")
    x = torch.randint(0, 2, shape, device="mps")
    for i in range(iterations):
        y = torch.unique(x)
        del y

        # Empty cache and collect garbage to make sure
        torch.mps.empty_cache()
        gc.collect()

        if i % 10 == 0:
            print(
                f"Iter {i}: Driver Allocated Memory: {torch.mps.driver_allocated_memory() / (1024**2):.2f}MB, Current Allocated Memory: {torch.mps.current_allocated_memory() / (1024**2):.2f}MB"
            )

    # Test 2: torch.sort (comparison)
    print("\nTest 2: torch.sort")
    for i in range(iterations):
        y = torch.sort(x)[0]
        del y
        # Empty cache and collect garbage to make sure
        torch.mps.empty_cache()
        gc.collect()

        if i % 10 == 0:
            print(
                f"Iter {i}: Driver memory: {torch.mps.driver_allocated_memory() / (1024**2):.2f}MB, Current memory: {torch.mps.current_allocated_memory() / (1024**2):.2f}MB"
            )


test_operations(iterations=100, shape=(2000, 10))