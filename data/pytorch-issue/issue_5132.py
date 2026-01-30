import torch.nn as nn

# create lstmcell module
lstm = nn.LSTM(embed_size * 2, hidden_size)

# transpose from batch first to batch second for recurrent layers
x = x.transpose(0, 1).contiguous()
output, (h_t, c_t) = lstm(x)

# create lstmcell module
lstm = nn.LSTMCell(embed_size * 2, hidden_size)

# initialize h,c outside for loop
h_t = Variable(x.data.new().resize_as_(x[:, 0, :].data).zero_())
c_t = Variable(x.data.new().resize_as_(x[:, 0, :].data).zero_())

# loop over time steps
for time_step in range(x.size(1)):
    x_t = x[:, time_step, :]
    (h_t, c_t) = lstm(x_t, (h_t, c_t))