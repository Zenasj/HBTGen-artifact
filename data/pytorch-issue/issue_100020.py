import torch

torch._logging.set_logs(
  dynamo=logging.DEBUG,
  bytecode=False,
)