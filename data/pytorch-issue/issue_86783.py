import torch.nn as nn

import cv2
import torch

def main():
    conv2d = torch.nn.Conv2d(3, 1, 1).cuda()
    input = torch.ones((1, 3, 32, 32)).cuda()

    conv2d.eval()
    with torch.no_grad():
        conv2d(input)

    print('after forward')
    cv2.waitKey(5000)

    conv2d.cpu()
    input.cpu()
    del conv2d, input
    torch.cuda.empty_cache()

    print('free memory')
    cv2.waitKey(5000)
    return

if __name__ == "__main__":
    main()