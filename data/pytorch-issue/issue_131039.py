import numpy as np
A = np.array([-1.4266, -0.2079, -0.9915, 0.0182, 1.1049])
print(A)
# (A + 1).pow(1 / 13) - 1
result = np.power((A + 1), 1/13) - 1
print(result)