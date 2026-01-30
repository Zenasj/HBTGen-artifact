import numpy as np
import time
import torch
def test():
    anchors = torch.ones((1000, 4), dtype=torch.float32).cuda()
    preds = torch.ones((1000, 4), dtype=torch.float32).cuda()

    #anchors = torch.ones((1000000, 4), dtype=torch.float32).cuda()
    #preds = torch.ones((1000000, 4), dtype=torch.float32).cuda()

    img_shape = (1080, 1496, 3)
    torch.cuda.synchronize()
    s = time.time()
    for i in range(50 * 5):
        # delta2bbox(anchors, preds, [0, 0, 0, 0], [1,1,1,1], img_shape)
        a = anchors * preds * anchors * preds
    torch.cuda.synchronize()
    print("delta2bbox is:", time.time() - s)

test()