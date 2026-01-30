import torch
import torch.nn as nn
import math

fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
# generate sqrt(3) in complicated way: sqrt(2 / (1 + 5)) = sqrt(1/3)
gain = torch.nn.init.calculate_gain("leaky_relu", math.sqrt(5)) 
# sqrt(3) b/c uniform gets rid of the sqrt(1/3) again
bound = math.sqrt(3) * gain / math.sqrt(fan_in) 
nn.init.uniform_(weight, -bound, bound)

fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
bound = 1 / math.sqrt(fan_in)
nn.init.uniform_(weight, -bound, bound)