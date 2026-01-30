from __future__ import print_function
import torch

import psutil, os
p = psutil.Process( os.getpid() )
for dll in p.memory_maps():
    print(dll.path)

import psutil, os
p = psutil.Process( os.getpid() )
for dll in p.memory_maps():
  print(dll.path)