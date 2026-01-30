import torch
print(f"torch version: " + torch.__version__)
print("\n\n--WITH CLONE--\n")
x = torch.arange(8).reshape(2, 4)
print(f"x: {x}")

y = x.clone().reshape(x.shape[0], 2, x.shape[1] // 2)
y[:, 1] = y[:, 1].flip((1, ))

x = torch.cat(x.unsqueeze(1).split(x.shape[1] // 2, dim=2), dim=1)
x[:, 1] = x[:, 1].flip((1, ))

print("\ny and x ARE " + ("" if torch.all(y == x) else "NOT ") + "equal\n")
print(f"y:\n {y}, \nx: \n{x}")

print("\n\n--WITHOUT CLONE--\n")
x = torch.arange(8).reshape(2, 4)
print(f"x: {x}")

y = x.reshape(x.shape[0], 2, x.shape[1] // 2)
y[:, 1] = y[:, 1].flip((1, ))

x = torch.cat(x.unsqueeze(1).split(x.shape[1] // 2, dim=2), dim=1)
x[:, 1] = x[:, 1].flip((1, ))

print("\ny and x ARE " + ("" if torch.all(y == x) else "NOT ") + "equal\n")
print(f"y:\n {y}, \nx: \n{x}")