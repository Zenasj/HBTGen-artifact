from tools.generate_torch_version import get_torch_version

# omitted

version = get_torch_version()

def get_torch_version(sha=None):
    pytorch_root = Path(__file__).parent.parent
    version = open('version.txt', 'r').read().strip()

# omitted