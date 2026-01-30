import torch

grid_data_variable = Variable(torch.rand(batch_size, gh, gw, d, nchans), requires_grad=True).cuda()
guide_data_variable = Variable(torch.rand(batch_size, h, w), requires_grad=True).cuda()

grid_data_variable = Variable(torch.rand(batch_size, gh, gw, d, nchans).cuda(), requires_grad=True)
guide_data_variable = Variable(torch.rand(batch_size, h, w).cuda(), requires_grad=True)

grid_data_variable

guide_data_variable

True

requires_grad

# Your first line can be separated into these two lines
user_grid_data_variable = Variable(torch.rand(batch_size, gh, gw, d, nchans), requires_grad=True)
grid_data_variable = user_grid_data_variable.cuda()
# user_grid_data_variable will have gradients properly computed
# grid_data_variable  will not have gradients as it is not user created, it has been created by the `.cuda()` operation