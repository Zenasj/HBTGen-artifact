#!/usr/bin/env python3
import sys
import torch


def _main():
    if len(sys.argv) != 3:
        print(f'Usage sys.argv[0] <Tensor1> <Tensor2>')
        sys.exit(1)

    tensor1 = torch.load(sys.argv[1])
    tensor2 = torch.load(sys.argv[2])

    torch.testing.assert_allclose(tensor1, tensor2)


if __name__ == '__main__':
    _main()