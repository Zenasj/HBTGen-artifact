var_delta = m_t / (K.sqrt(v_t) + epsilon_t)
var_update = state_ops.assign_sub(var, lr_t * var_delta, use_locking=self._use_locking)

X1 = math_ops.sqrt(v_t)
X2 = K.sqrt(v_t)
Y1 = m_t / (math_ops.sqrt(v_t) + epsilon_t)
Y2 = m_t / (K.sqrt(v_t) + epsilon_t)

print(X1);       print(X2);       print()
print(Y1);       print(Y2);       print()
print(type(X1)); print(type(X2)); print()
print(type(Y1)); print(type(Y2)); print()
print("type(ref) =", type(var))

from tensorflow.python.framework.ops import Tensor, EagerTensor
print(Tensor.__doc__) # OK
print(EagerTensor.__doc__) # returns None

from inspect import getsource
print(getsource(EagerTensor)) # OSError: could not find class definition

EagerTensor = c_api.TFE_Py_InitEagerTensor(_EagerTensorBase)