import torch.nn as nn

import torch
import torch.nn.functional as F

def main():
    x = torch.ones(10, 1).to(torch.device('mps'))
    x = F.softplus(x)
    

if __name__ == "__main__":
    main()