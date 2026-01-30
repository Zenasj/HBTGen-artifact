import torch

torch.hub.set_dir("~/example_dir/")
torch.hub.get_dir()  # Returns the string "~/example_dir/"

os.makedirs("~/example_dir/")  # Will create a directory in the current directory called ./~/example_dir

def set_dir(d):
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.
    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = os.path.expand_dirs(d)