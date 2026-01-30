import torch.nn as nn

import torch
batch_size = 16
lstm_hidden_size = 32
input_size = 8
output_size = 8
device = torch.device('cpu')
lstm_cell = torch.nn.LSTMCell(input_size + output_size, lstm_hidden_size).to(device)
predict_halt = torch.nn.Sequential(torch.nn.Linear(lstm_hidden_size, 1),
                                   torch.nn.Sigmoid()).to(device)
kwargs = {'device': device}
next_lstm_hidden_state = torch.zeros(batch_size, lstm_hidden_size, **kwargs)
next_lstm_cell_state = torch.zeros(batch_size, lstm_hidden_size, **kwargs)
lstm_hidden_state = torch.zeros(batch_size, lstm_hidden_size, **kwargs)
lstm_cell_state = torch.zeros(batch_size, lstm_hidden_size, **kwargs)
halt_cumulative = torch.zeros(batch_size, **kwargs)
halted = torch.zeros(batch_size, **kwargs).byte()
# Break on exceeding output or input
for i in range(10):
    input_ = torch.rand(batch_size, input_size, requires_grad=True, **kwargs)
    output = torch.rand(batch_size, output_size, requires_grad=True, **kwargs)
    # [batch_size, input_size + output_size]
    # Embedding of the current output and input state
    lstm_input = torch.cat((input_, output), dim=1)
    # lstm_hidden_state [batch_size, lstm_hidden_size]
    # lstm_cell_state [batch_size, lstm_hidden_size]
    lstm_hidden_state, lstm_cell_state = lstm_cell(lstm_input, (lstm_hidden_state, lstm_cell_state))
    #
    halt_probability = predict_halt(lstm_hidden_state)
    remainder = 1 - halt_cumulative
    halt_cumulative += torch.min(halt_probability.squeeze(1), remainder)
    halted = halt_cumulative.ge(1 - 0.01)
    # Update hidden state with remainder for ``halted`` sequences
    next_lstm_hidden_state[halted] = (lstm_hidden_state * remainder.unsqueeze(1))[halted]
    next_lstm_cell_state[halted] = (lstm_cell_state * remainder.unsqueeze(1))[halted]
    # Update hidden state with ``halt_probability`` for other sequences
    next_lstm_hidden_state[~halted] = (lstm_hidden_state * halt_probability)[~halted]
    next_lstm_cell_state[~halted] = (lstm_cell_state * halt_probability)[~halted]
    # Reset state for ``halted`` sequences, next hidden state is an accumulation from the
    # past states

    lstm_hidden_state[halted] = next_lstm_hidden_state[halted]
    lstm_cell_state[halted] = next_lstm_cell_state[halted]
    halt_cumulative = halt_cumulative.masked_fill(halted, 0)
    next_lstm_hidden_state = next_lstm_hidden_state.masked_fill(halted.unsqueeze(1), 0)
    next_lstm_cell_state = next_lstm_cell_state.masked_fill(halted.unsqueeze(1), 0)
lstm_hidden_state.sum().backward()