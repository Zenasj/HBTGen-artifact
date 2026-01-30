import torch
concat_list = []

concat_list.append(torch.ones((6500,1024*512), dtype=torch.uint8))
concat_list.append(torch.ones((4500,1024*512), dtype=torch.uint8))

ccat = torch.cat(concat_list)