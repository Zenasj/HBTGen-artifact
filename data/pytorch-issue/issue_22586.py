import torch
import torch.nn as nn
import math

if __name__ == "__main__":
    batch_size = 128
    temperature = 5.0
    theta = torch.FloatTensor([1.753356814384460449,1.898535370826721191,0.6992630958557128906,
                                0.2227068245410919189,0.6384450793266296387,1.431323885917663574,
                                -0.05012089386582374573, -0.06672633439302444458])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_gpu = theta.repeat(batch_size, 1).to(device)
    max_num = 1000000
    nan_num = 0
    for i in range(max_num):
        weight = nn.functional.gumbel_softmax(t_gpu, temperature)
        if math.isnan(torch.sum(weight)):
            nan_num+=1
    print("GPU: nan {:.3f}% probability happen, tot {}".format(100.0 * nan_num / max_num, nan_num))
    nan_num = 0
    t_cpu = theta.repeat(batch_size, 1)
    for i in range(max_num):
        weight = nn.functional.gumbel_softmax(t_cpu, temperature)
        if math.isnan(torch.sum(weight)):
            nan_num+=1
    print("CPU: nan {:.3f}% probability happen, tot {}".format(100.0 * nan_num / max_num, nan_num))