import torch

input_len = Variable(torch.from_numpy(input_len[input_sort_id])).int().cuda()
feat_mat = feats[th.LongTensor(input_sort_id)]
feat_mat = Variable(feat_mat.cuda())