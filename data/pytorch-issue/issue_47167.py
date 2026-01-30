for name, param in m.named_parameters():
  if not param.grad:
    print(f"detected unused parameter: {name}")