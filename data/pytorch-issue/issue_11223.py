import torch
import torch.nn as nn
import random

def _seg_max(data, seg, n_seg):
    'Perform segment max'
    _, sorted_id = torch.sort(data)
    sorted_seg = seg[sorted_id].detach()
    sorted_data = data[sorted_id].detach()
    seg_max = torch.zeros((n_seg,), device=data.device).index_copy(0, sorted_seg, sorted_data).detach()
    return seg_max

def random_seg(n_ele, n_seg):
    'Create a random segment idx list'
    seg_list = [random.randint(0, n_seg - 1) for _ in range(n_ele - n_seg)] + list(range(n_seg))
    random.shuffle(seg_list)
    return seg_list

def run(n_ele, n_seg, device):
    print('Perform `reduce max` on {}-dim vector into {} segments on {}'.format(n_ele, n_seg, device))
    with torch.no_grad():
        for _ in range(10000):
            data = torch.FloatTensor(n_ele).to(device)
            nn.init.uniform_(data)
            segment_ids = torch.LongTensor(random_seg(n_ele, n_seg)).to(device)
            seg_max = _seg_max(data, segment_ids, n_seg)
            a = seg_max.max().item()
            b = data.max().item()
            assert b == a, "{} != {}".format(a, b)
    print(' -- Passed.')


gpu = torch.device('cuda')
cpu = torch.device('cpu')

run(n_ele=40, n_seg=40, device=cpu)
run(n_ele=40, n_seg=10, device=cpu)
run(n_ele=40, n_seg=40, device=gpu)
run(n_ele=40, n_seg=10, device=gpu)