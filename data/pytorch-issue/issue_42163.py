import hashlib
import torch


def get_sha256_hash(file: str, chunk_size: int = 4096) -> str:
    hasher = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


file = "tensor.pt"
hashes = []
for _ in range(5):
    obj = torch.ones(1)
    torch.save(obj, file)
    hashes.append(get_sha256_hash(file)[:8])
    del obj

hash = hashes[0]
assert all(other == hash for other in hashes[1:])
print(hash)