import torch

def test_enumarete(a: str):
  res: List[int] = []
  for i, _ in enumerate(a, start=1):
    res.append(i)
  return res

print(test_enumarete("abc"))
# print [1, 2, 3]

script_method = torch.jit.script(test_enumarete)
print(script_method("abc"))
# print [0, 1, 2]