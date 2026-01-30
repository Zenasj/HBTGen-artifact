import numpy as np
import torch


def numpy_example() -> None:
    xs_f = np.array([567.5, 568.5, 567.5, 568.5, 567.5, 568.5])
    scores = np.array([73.5, 73.5, 100.5, 98.5, 41.5, 37.5])
    nscores = scores / np.sum(scores)

    print(np.sum(xs_f * scores) / np.sum(scores))                           # 567.9929411764706
    print(np.sum(xs_f * nscores))                                           # 567.9929411764705
    print(xs_f.T @ nscores)                                                 # 567.9929411764706

    xs_f_32 = xs_f.astype(np.float32)
    scores_32 = scores.astype(np.float32)
    nscores_32 = nscores.astype(np.float32)

    print((np.sum(xs_f_32 * scores_32) / np.sum(scores_32)).item())         # 567.992919921875
    print(np.sum(xs_f_32 * nscores_32).item())                              # 567.992919921875
    print((xs_f_32.T @ nscores_32).item())                                  # 567.992919921875


def torch_example() -> None:
    xs_f = torch.tensor([567.5, 568.5, 567.5, 568.5, 567.5, 568.5], dtype=torch.float64)
    scores = torch.tensor([73.5, 73.5, 100.5, 98.5, 41.5, 37.5], dtype=torch.float64)
    nscores = scores / torch.sum(scores)

    print((torch.sum(xs_f * scores) / torch.sum(scores)).item())            # 567.9929411764706
    print(torch.sum(xs_f * nscores).item())                                 # 567.9929411764706
    print((xs_f @ nscores).item())                                          # 567.9929411764706

    xs_f_32 = xs_f.type(torch.float32)
    scores_32 = scores.type(torch.float32)
    nscores_32 = nscores.type(torch.float32)

    print((torch.sum(xs_f_32 * scores_32) / torch.sum(scores_32)).item())   # 567.992919921875
    print(torch.sum(xs_f_32 * nscores_32).item())                           # 567.9929809570312 <--

    p = torch.tensor(0, dtype=torch.float32)
    for i in range(len(xs_f_32)):
        p += xs_f_32[i] * nscores_32[i]

    print(p.item())                                                         # 567.992919921875

    print((xs_f_32 @ nscores_32).item())                                    # 567.9929809570312 <--


if __name__ == "__main__":
    numpy_example()
    torch_example()