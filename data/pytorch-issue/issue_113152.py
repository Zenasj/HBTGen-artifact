import torch.nn as nn

# Without names.
model = nn.Sequential(
   hello_module,
   world_module,
)

# With names.
from collections import OrderedDict

model = nn.Sequential(
    OrderedDict(
        [
            ('hello', hello_module),
            ('world', world_module),
        ]
    )
)

# With names.
model = nn.Sequential(
    ('hello', hello_module),
    ('world', world_module),
)