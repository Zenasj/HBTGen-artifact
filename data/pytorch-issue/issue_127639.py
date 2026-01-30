import concurrent.futures
import hashlib
import time

# Importing torch before the fork causes slowdown:
import torch

def task(n):
    # No slowdown if we import torch here instead:
    # import torch

    path = "/home/slarsen/.conda/envs/pytorch-3.11/lib/python3.11/site-packages/triton/_C/libtriton.so"

    libtriton_hash = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)

    return True


def main():
    start = time.time()

    num_workers = 32
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(task, n): n for n in range(num_workers)}
        results = {}
        for future in concurrent.futures.as_completed(futures.keys()):
            results[futures[future]] = future.result()
        assert(len(results) == num_workers)
        #print(results)

    print(f"{time.time() - start:.2f}")


if __name__ == "__main__":
    main()