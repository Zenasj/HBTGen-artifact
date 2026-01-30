import torch

from lenses import bind

def nflatten(self, **kwargs):
    for name,olds in kwargs.items():
        olds = tuple(bind(olds).Recur(str).collect())
        self = self.align_to(..., *olds).flatten(olds, name) if olds else self.rename(None).unsqueeze(-1).rename(*self.names, name)
    return self

def nunflatten(self, **kwargs):
    for name,news in kwargs.items():
        news = tuple(bind(news).Each().collect())
        self = self.unflatten(name, news) if news else self.squeeze(name)
    return self

torch.Tensor.nflatten = nflatten
torch.Tensor.nunflatten = nunflatten