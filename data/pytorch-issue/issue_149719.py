import torch

def get_best_device(priority = ("cuda", "mps", "cpu")):
    """Returns the best available device from a priority list.
    Args:
        priority_list (list): List of device names in decreasing priority.
    Returns:
        torch.device: The best available device.
    Raises:
        ValueError: If no suitable device is found from the priority list.
    """
    for device_name in priority_list:
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device_name == "cpu":
            return torch.device("cpu")
    raise ValueError("No suitable device found from the priority list.")