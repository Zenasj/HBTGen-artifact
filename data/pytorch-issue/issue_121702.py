import torch

cpp
ASSERT_TRUE(
      THPException_OutOfMemoryError = PyErr_NewExceptionWithDoc(
          "torch.OutOfMemoryError",
          "Exception raised when device is out of memory",
          PyExc_RuntimeError,
          nullptr));