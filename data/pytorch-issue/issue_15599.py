# torch.rand(B, T, F, dtype=torch.float32)  # B=16, T=10, F=16 (input_size + output_size)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm_cell = nn.LSTMCell(16, 32)  # input_size=8+8, hidden_size=32
        self.predict_halt = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        T, B, _ = x.size()
        device = x.device
        hidden_size = 32

        # Initialize states
        lstm_hidden = torch.zeros(B, hidden_size, device=device)
        lstm_cell = torch.zeros(B, hidden_size, device=device)
        next_lstm_hidden = torch.zeros_like(lstm_hidden)
        next_lstm_cell = torch.zeros_like(lstm_cell)
        halt_cumulative = torch.zeros(B, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(T):
            current_input = x[t]

            # Run LSTMCell
            new_h, new_c = self.lstm_cell(current_input, (lstm_hidden, lstm_cell))

            # Compute halt probability
            halt_prob = self.predict_halt(new_h).squeeze(1)
            remainder = 1 - halt_cumulative
            halt_cumulative += torch.min(halt_prob, remainder)
            halted = halt_cumulative.ge(1 - 0.01)

            remainder_unsq = remainder.unsqueeze(1)

            # Compute next states
            term_halted_h = new_h * remainder_unsq
            term_halted_c = new_c * remainder_unsq
            term_nonhalted_h = new_h * halt_prob.unsqueeze(1)
            term_nonhalted_c = new_c * halt_prob.unsqueeze(1)

            next_lstm_hidden = torch.where(
                halted.unsqueeze(1),
                term_halted_h,
                term_nonhalted_h
            )
            next_lstm_cell = torch.where(
                halted.unsqueeze(1),
                term_halted_c,
                term_nonhalted_c
            )

            # Update current states with next states for halted
            lstm_hidden = next_lstm_hidden
            lstm_cell = next_lstm_cell

            # Reset for halted sequences
            halt_cumulative = halt_cumulative.masked_fill(halted, 0)
            next_lstm_hidden = next_lstm_hidden.masked_fill(halted.unsqueeze(1), 0)
            next_lstm_cell = next_lstm_cell.masked_fill(halted.unsqueeze(1), 0)

        return lstm_hidden

def my_model_function():
    return MyModel()

def GetInput():
    B = 16
    T = 10
    F = 16  # 8 + 8
    return torch.rand(T, B, F, requires_grad=True, dtype=torch.float32)

