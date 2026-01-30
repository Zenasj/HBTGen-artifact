import torch
from torch.profiler import profile, ProfilerActivity

def check_cupti_enabled():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return False
    
    # Create a simple CUDA tensor
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")

    try:
        # Use PyTorch profiler to perform a basic check
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            z = x @ y  # Simple CUDA operation
        
        # Print profiling results
        print("CUPTI is enabled and profiling works.")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return True
    except RuntimeError as e:
        # If profiling fails, CUPTI is likely not set up correctly
        print("Error: CUPTI might not be enabled or accessible.")
        print(f"Details: {e}")
        return False

if __name__ == "__main__":
    if check_cupti_enabled():
        print("CUPTI is properly configured in PyTorch.")
    else:
        print("CUPTI is not configured correctly. Check your CUDA installation.")