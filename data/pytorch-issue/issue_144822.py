import torch
from datetime import datetime

batch_size = 1024 * 8000
final_num = 2147483647
num_batches = final_num // batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


# @torch.compile
def bench():
    final_total = torch.tensor(0, dtype=torch.int32, device=device)

    for i in range(num_batches):
        batch = torch.arange(
            i * batch_size, (i + 1) * batch_size, dtype=torch.int32, device=device
        )
        batch_sum = torch.sum(batch % 2)
        final_total += batch_sum

    return final_total


a = datetime.now()
final_total = bench()
b = datetime.now()

print(f"Final Total: {final_total.item()}, {b-a} seconds")

bench()

a = datetime.now()
final_total = bench()
b = datetime.now()