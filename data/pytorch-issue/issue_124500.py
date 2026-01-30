import torch
import tempfile
import json
from torch.profiler import ExecutionTraceObserver

def fn():
  fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
  fp.close()

  et = ExecutionTraceObserver()
  observer = et.register_callback(fp.name)

  with torch.profiler.profile(execution_trace_observer=observer) as prof:
    # OK if we graph break here
    torch.tensor([0, 2, 4, 6, 8])

  with open(fp.name) as f:
    et_graph = json.load(f)

out = torch.compile(fn, backend="eager")()