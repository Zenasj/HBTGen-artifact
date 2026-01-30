import torch
import torch.nn as nn

CUDA = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if CUDA else "cpu")
embed_gpu = nn.Embedding.from_pretrained(torch.rand((2, 3)), freeze=True).to(DEVICE)