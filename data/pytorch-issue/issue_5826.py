num_calls = [0]
def feval(x):
  num_calls[0] += 1
  net.updateOutput(x.cuda())
  grad = net.updateGradInput(x.cuda(), dy.cuda())
  loss = 0
  for mod in content_losses:
    loss = loss + mod.loss
  for mod in style_losses:
    loss = loss + mod.loss
  return loss, grad.view(grad.nelement())

optim_state = None
if params.optimizer == 'lbfgs':
  optim_state = {
    "maxIter": params.num_iterations,
    "verbose": True,
    "tolX": -1,
    "tolFun": -1,
  }
  if params.lbfgs_num_correction > 0:
    optim_state.nCorrection = params.lbfgs_num_correction
elif params.optimizer == 'adam':
    optim_state = {
      "learningRate": params.learning_rate,
    }

# Run optimization.
if params.optimizer == 'lbfgs':
  print("Running optimization with L-BFGS")
  x, losses = optim.lbfgs(feval, img, optim_state)
elif params.optimizer == 'adam':
  print("Running optimization with ADAM")
  for t in xrange(params.num_iterations):
    x, losses = optim.adam(feval, img, optim_state)