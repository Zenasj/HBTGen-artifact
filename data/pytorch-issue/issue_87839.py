import torch.nn as nn
import pickle
m = nn.Linear(32, 32).to('mps')
inband = pickle.dumps(m.state_dict())