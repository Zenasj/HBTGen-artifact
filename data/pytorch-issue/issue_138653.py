import numpy as np
from scipy.special import zeta

x_nan = np.nan
q_nan = np.nan

result1 = zeta(x_nan, 1)  
result2 = zeta(1, q_nan)  

print(result1)  # nan
print(result2)  # inf

special.zeta(1, np.nan)
# inf

import math

inf = math.inf
nan = math.nan

math.pow(nan, 0)
# 1.0

math.hypot(inf, nan)
# inf

math.hypot(-inf, nan)
# inf

math.pow(1, nan)
# 1.0