import random

import numpy
import time
from torchDynamo import *
# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------
dtype = "float32"

# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------

def evaluator(s, inputs, num):
  all_time = []
  for i in range(num):
      torch.cuda.synchronize()
      start = time.time()
      result = s(inputs)
      torch.cuda.synchronize()
      end = time.time()
      elapsed_time = end - start
      all_time.append(elapsed_time)

  # 计算时间的平均值
  average_time = sum(all_time) / num

  return average_time


def evaluate_operation(s, inputs, optimization, log):
  """Evaluate operation correctness and print the performance information.
  Args:
    s: The schedule to be built.
    vars: The argument lists to the function.
    target: The target and option of the compilation.
    inputs: The input tensors.
    standard: The standard result for correctness evaluation.
    optimization: The name of the optimization.
    log: The log list.
  """
  mean_time = evaluator(s, inputs, 1)
  log.append((optimization, mean_time))


def report_performance(log):
  """Convert the log into a performance table.
  Args:
    log: The log list.
  """
  baseline = log[-1][1]
  header = "Benchmark".ljust(20) + "\t" + "Time".rjust(
      10) + "\t" + "SpeedUp".rjust(10)
  split_line = "-" * 50
  print(split_line)
  print(header)
  print(split_line)
  for result in log:
    formatted_time = "{:.2f}".format(result[1])
    formatted_performance = "{:.2f}".format(baseline / result[1])
    print("\033[32m%s\033[0m\t\033[33m%s\033[0m\t\033[34m%s\033[0m" %
          (result[0].ljust(20), str(formatted_time + " ms").rjust(10),
           str(formatted_performance).rjust(10)))


def main():
  # ----------------------------------------------------------------------------
  # Initialization and Baseline
  # ----------------------------------------------------------------------------
  # Initialize the log list.
  log = []

  # Generate random tensor for testing.
  size = (512, 64, 3)
  c, n, k, p, s = size[0], size[0], size[1], size[2], 1
  oc, ic, n, k, p, s = size[0], size[0], size[1], size[2], 1, 1
  data,weight, out = get_conv_data_torch(c, n, k, p, s)


  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.

  s_1 = conv_torch(data, out, k, p, s)
  evaluate_operation(s_1,
                     inputs=data,
                     optimization="torch_conv_default",
                     log=log)


  s_2 = conv_compiled(data, out, k, p, s)
  evaluate_operation(s_2,
                     inputs=data,
                     optimization="torch_conv_dynamo",
                     log=log)

 

  report_performance(log)


if __name__ == "__main__":
  main()

import torch
import torch.nn as nn
import numpy as np


def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p)//s + 1



def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
    tensor with the shapes specified by input arguments.

    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype('float32')
    weight = np.random.normal(size=(oc, ic, k, k)).astype('float32')
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype='float32')
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out


def conv_torch(data, out, k, p, s):
    f = nn.Conv2d(data.shape[1], out.shape[1], kernel_size=k, stride=s, padding=p)
    return f

def conv_compiled(data, out, k, p, s):
    f = nn.Conv2d(data.shape[1], out.shape[1], kernel_size=k, stride=s, padding=p)
    f_s = torch.compile(f)
    return f_s

def get_conv_data_torch(c, n, k, p, s):
    data, weight, out = get_conv_data(c, c, n, k, p, s,lambda x: torch.from_numpy(x))
    data = data.unsqueeze(0)  # 在第0个维度前添加一个新维度
    out = out.unsqueeze(0)
    return data, weight, out