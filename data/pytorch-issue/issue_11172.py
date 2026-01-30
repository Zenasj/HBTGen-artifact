import torch

import pickle
import torch

if __name__ == '__main__':
    with open('test', 'wb') as f:
        pickle.dump(torch.float32, f)
    print('pickle success')