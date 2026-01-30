import torch
root = []                
root[:] = [root, root, None, None] 
def test_bug():
    return root

fn = test_bug
fn = torch.compile(fn)

print(fn())