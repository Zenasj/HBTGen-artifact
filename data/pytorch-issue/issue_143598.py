py
import time
from concurrent.futures import ThreadPoolExecutor
from torch import distributed as dist

def run():
    store = dist.TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )

    # this sleep is required to trigger the crash
    time.sleep(0.1)
    del store

futures = []
with ThreadPoolExecutor(
    max_workers=100,
) as executor:
    for i in range(100000):
        print(i)
        futures.append(executor.submit(run))
        if len(futures) > 100:
            futures.pop(0).result()