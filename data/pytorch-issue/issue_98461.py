opt_fn = compile_train_step(fn)
loss = opt_fn(inputs, optimizer, ...)