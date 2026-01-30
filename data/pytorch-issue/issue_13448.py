import torch

Ns = [10, 100, 1000, 10000]

for N in Ns:

    nans_32_bit = (torch.zeros(N) / 0)
    nans_64_bit = (nans_32_bit.type(torch.DoubleTensor))

    print('Input: tensor of size N={} containing only NaN'.format(N))

    post_sigmoid_32_bit = torch.sigmoid(nans_32_bit)
    post_sigmoid_64_bit = torch.sigmoid(nans_64_bit)

    proportion_nan_32_bit = torch.mean(torch.isnan(post_sigmoid_32_bit).type(torch.FloatTensor))
    proportion_nan_64_bit = torch.mean(torch.isnan(post_sigmoid_64_bit).type(torch.FloatTensor))

    print('After sigmoid:')
    print('Proportion NaN in result with 32 bit input: {:.3f}'.format(proportion_nan_32_bit))
    print('Proportion NaN in result with 64 bit input: {:.3f}\n'.format(proportion_nan_64_bit))