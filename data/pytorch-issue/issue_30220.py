import torch.nn.functional as F

import os
import numpy as np
import torchvision.transforms.functional as F

while True:
    a = bytearray(os.urandom(104 * 104 * 3))
    im = np.array(a).reshape((104, 104, 3))
    im = F.to_tensor(im)
    im = F.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im = im.view(1, *im.size())
    print(im.shape)

import os
import numpy as np
import torchvision.transforms.functional as F

while True:
    a = bytearray(os.urandom(105 * 105  * 3))
    im = np.array(a).reshape((105 , 105 , 3))
    im = F.to_tensor(im)
    im = F.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im = im.view(1, *im.size())
    print(im.shape)