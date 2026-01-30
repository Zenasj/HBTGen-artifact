import io
import torch
import torch.nn as nn

example_model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
)

buffer = io.BytesIO()
torch.save(example_model.state_dict(), buffer)
loaded = torch.load(buffer)

import io
import torch
import torch.nn as nn

example_model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
)

buffer = io.BytesIO()
torch.save(example_model.state_dict(), buffer)

# Set the stream position to the beginning
buffer.seek(0)

loaded = torch.load(buffer)

# loaded = OrderedDict([('0.weight', ...