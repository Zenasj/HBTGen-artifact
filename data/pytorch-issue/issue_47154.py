import torch


def diff(x1, x2):
    ans1 = x1 @ x2
    ans2 = []
    for i in range(x1.size(0)):
        ans2.append(x1[i] @ x2[i])
    ans2 = torch.stack(ans2)

    print('diff_mean =', (ans2 - ans1).abs().mean())
    print('diff_max =', (ans2 - ans1).abs().max())
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    x1 = torch.randn(32, 512, 64).cuda() * 1e4
    x2 = torch.randn(32, 64, 128).cuda() * 1e4

    diff(x1[:, :1, :], x2)
    diff(x1[:, :128 * 1, :], x2)
    diff(x1[:, :128 * 2, :], x2)
    diff(x1[:, :128 * 3, :], x2)
    diff(x1[:, :128 * 4, :], x2)
    diff(x1, x2)