import torch.nn.functional as F

def cos_sim(v1,v2):
    return F.cosine_similarity(v1.unsqueeze(0),v2.unsqueeze(0))

vv1 = tensor(list([float(i) for i in range(84)])).unsqueeze(0)
vv2 = tensor(list([float(i) for i in range(84)])).unsqueeze(0)

F.cosine_similarity(vv1,vv2).item()

1.0000001192092896

x / (sqrt(x) * sqrt(x))  # bad

x / sqrt(x * x)  # good