import torch
import torch.nn as nn

lstm = torch.nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state

filepath = "lstm.pt"
lstm.load_state_dict(torch.load(filepath),strict=True) 
print(lstm.weight_ih_l0)
tensor([[ 1.4603e-25, -1.5686e-21,  1.2081e-19],
        [-2.6189e-12,  1.5163e+28, -3.3969e+12],
        [ 1.7073e-38,  6.6708e+25, -4.6704e-39],
        [-1.6697e-03, -2.6238e-19, -1.3629e+16],
        [-6.7119e-16, -1.0742e+04, -6.3514e-39],
        [ 5.6304e+19,  4.6716e+20, -1.9736e+27],
        [-1.8113e+29, -6.7717e+25, -9.6834e+26],
        [-1.0278e+37, -2.5245e-12, -9.8826e+10],
        [ 3.4006e-17,  1.4774e-19,  1.0029e-10],
        [-8.1701e+28,  6.6931e-05,  9.2278e-03],
        [-7.8279e+19,  5.4253e-05,  1.5379e+03],
        [-8.6688e+24,  5.1006e-06,  1.1713e-11]], requires_grad=True)

print("Byteswap state_dict")
for param_tensor in lstm.state_dict():
    print(param_tensor, "\t", lstm.state_dict()[param_tensor].size())
    print(lstm.state_dict()[param_tensor])
    updated = lstm.state_dict()[param_tensor].cpu().detach().numpy().byteswap()
    lstm.state_dict()[param_tensor][:] = torch.Tensor(updated)
    print(lstm.state_dict()[param_tensor])