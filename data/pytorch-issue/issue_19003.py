optimizer = optim.Adam(params_to_update, lr=5e-4)
scheduler = CyclicLR(optimizer, base_lr=5e-6, max_lr=5e-2, cycle_momentum=False, step_size_up=2500)

cycle_momentum=False