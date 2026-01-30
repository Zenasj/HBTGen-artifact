import torch

import threading
from torch._functorch.vmap import lazy_load_decompositions

threads = []
for i in range(10000):
    thread = threading.Thread(target=lazy_load_decompositions)
    threads.append(thread)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()