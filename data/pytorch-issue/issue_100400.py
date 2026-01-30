import torch

import urllib.request
try:
    for i in range(400):
       x=urllib.request.urlopen('https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth')
except Exception as e:
    e1 = e
    print(e1)
    import pdb
    pdb.set_trace()