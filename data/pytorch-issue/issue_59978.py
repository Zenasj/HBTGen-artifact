import torch

import tempfile
f = tempfile.NamedTemporaryFile()
cmd = f"python -m torch.distributed.launch --init_method=file://{f.name} --nproc_per_node=2 ..."

def pytest_xdist_worker_id():
    """
    Returns an int value of worker's numerical id under ``pytest-xdist``'s concurrent workers ``pytest -n N`` regime,
    or 0 if ``-n 1`` or ``pytest-xdist`` isn't being used.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker = re.sub(r"^gw", "", worker, 0, re.M)
    return int(worker)


def get_torch_dist_unique_port():
    """
    Returns a port number that can be fed to ``torch.distributed.launch``'s ``--master_port`` argument.

    Under ``pytest-xdist`` it adds a delta number based on a worker id so that concurrent tests don't try to use the
    same port at once.
    """
    port = 29500
    uniq_delta = pytest_xdist_worker_id()
    return port + uniq_delta