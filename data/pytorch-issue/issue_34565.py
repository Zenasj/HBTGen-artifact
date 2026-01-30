# test.py
import torch

with torch.autograd.profiler.emit_nvtx():
  w = torch.randn(2, 2).cuda().requires_grad_()
  b = torch.randn(2, 1).cuda().requires_grad_()
  x = torch.randn(2, 1).cuda()
  y = (w @ x + b).sum()
  y.backward()

# log.py
import sqlite3
import pandas as pd

with sqlite3.connect("cuda.prof") as conn:
  print(pd.read_sql_query("SELECT value FROM CUPTI_ACTIVITY_KIND_MARKER as markers INNER JOIN StringTable as names on markers.name = names._id_ WHERE markers.flags = 2", conn))