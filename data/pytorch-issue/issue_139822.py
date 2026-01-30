import torch.nn as nn

import torch

def test_fn():
  try:
      model = torch.nn.Sequential(
          torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
      )

      x = torch.randn(32, 10)
      y = model(x)
      return True
  except Exception as e:
      print(f"Model test failed: {str(e)}")
      return False

comp = torch.compile(options={"fx_graph_remote_cache": False})(test_fn)
comp()