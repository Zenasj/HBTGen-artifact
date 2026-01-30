optimizer = torch.optim.LBFGS(x, max_iter=200)
optimizer.step(closure)

optimizer = torch.optim.LBFGS(x, max_iter=20)
for epoch in range(10):
    optimizer.step(closure)

import pdb
import copy
import scipy.optimize
import torch
 
def rosenbrock(x):
    answer = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer
 
evalFunc = rosenbrock
 
x0s = [[76.0, 97.0, 20.0, 120.0, 0.01, 1e2], [76.0, 97.0, 20.0, 120.0, 0.01, 1e4]]
trueMins = [torch.ones(len(x0)) for x0 in x0s]
 
toleranceGrad = 1e-5
toleranceChange = 1e-9
xConvTol = 1e-4
lineSearchFn = "strong_wolfe"
maxIterOneEpoch = 1000
maxIterMultipleEpochs = 100
nEpochs = 10
assert maxIterOneEpoch==nEpochs*maxIterMultipleEpochs
 
def closure():
    optimizer.zero_grad()
    curEval = evalFunc(x=x[0])
    curEval.backward(retain_graph=True)
    return curEval
 
for x0, trueMin in zip(x0s, trueMins):
    print("Results for x0=", x0)
 
    xOneEpoch = torch.tensor(copy.deepcopy(x0))
    xOneEpoch.requires_grad = True
    x = [xOneEpoch]
    optimizer = torch.optim.LBFGS(x, max_iter=maxIterOneEpoch, line_search_fn=lineSearchFn, tolerance_grad=toleranceGrad, tolerance_change=toleranceChange)
    optimizer.step(closure)
    stateOneEpoch = optimizer.state[optimizer._params[0]]
    funcEvalsOneEpoch = stateOneEpoch["func_evals"]
    nIterOneEpoch = stateOneEpoch["n_iter"]
    print("\tResults for one epochs:")
    print("\t\tConverged: {}".format(torch.norm(xOneEpoch-trueMin, p=2)<xConvTol))
    print("\t\tFunction evaluations: {:d}".format(funcEvalsOneEpoch))
    print("\t\tIterations: {:d}\n".format(nIterOneEpoch))
 
    xMultipleEpochs = torch.tensor(copy.deepcopy(x0))
    xMultipleEpochs.requires_grad = True
    x = [xMultipleEpochs]
    optimizer = torch.optim.LBFGS(x, max_iter=maxIterMultipleEpochs, line_search_fn=lineSearchFn, tolerance_grad=toleranceGrad, tolerance_change=toleranceChange)
    for epoch in range(nEpochs):
        optimizer.step(closure)
    stateMultipleEpochs = optimizer.state[optimizer._params[0]]
    funcEvalsMultipleEpochs = stateMultipleEpochs["func_evals"]
    nIterMultipleEpochs = stateMultipleEpochs["n_iter"]
    print("\tResults for multiple epochs:")
    print("\t\tConverged: {}".format(torch.norm(xMultipleEpochs-trueMin, p=2)<xConvTol))
    print("\t\tFunction evaluations: {:d}".format(funcEvalsMultipleEpochs))
    print("\t\tIterations: {:d}\n".format(nIterMultipleEpochs))

import pdb
import copy
import scipy.optimize
import torch
import numpy as np
import jax.numpy as jnp
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)

def rosenbrock(x):
    answer = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

evalFunc = rosenbrock

x0 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e4]
ptTrueMin = torch.ones(len(x0))
jnpTrueMin = jnp.ones(len(x0))

toleranceGrad = 1e-5
toleranceChange = 1e-9
xConvTol = 1e-6
lineSearchFn = "strong_wolfe"
maxIter = 1000

def closure():
    optimizer.zero_grad()
    curEval = evalFunc(x=x[0])
    curEval.backward(retain_graph=True)
    return curEval

xPT = torch.tensor(copy.deepcopy(x0))
xPT.requires_grad = True
x = [xPT]
optimizer = torch.optim.LBFGS(x, max_iter=maxIter, line_search_fn=lineSearchFn, tolerance_grad=toleranceGrad, tolerance_change=toleranceChange)
optimizer.step(closure)

statePT = optimizer.state[optimizer._params[0]]
funcEvalsPT = statePT["func_evals"]
nIterPT = statePT["n_iter"]
print("Results for troch.optim:")
print("\tConverged: {}".format(torch.norm(xPT-ptTrueMin, p=2)<xConvTol))
print("\tFunction evaluations: {:d}".format(funcEvalsPT))
print("\tIterations: {:d}\n".format(nIterPT))

minimizeOptions = {'gtol': toleranceGrad, 'maxiter': maxIter}
jx0 = jnp.array(x0)
optimRes = jax.scipy.optimize.minimize(fun=evalFunc, x0=jx0, method='BFGS', options=minimizeOptions)

print("Results for jax.scipy.otpimize:")
print("\tConverged: {}".format(jnp.linalg.norm(optimRes.x-jnpTrueMin, ord=2)<xConvTol))
print("\tFunction evaluations: {:d}".format(optimRes.nfev))
print("\tIterations: {:d}\n".format(optimRes.nit))

import pdb
import time
import copy
import scipy.optimize
import torch
import numpy as np
import jax.numpy as jnp
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)

def rosenbrock(x):
    answer = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

evalFunc = rosenbrock
jEvalFunc = jax.jit(evalFunc)

x0_1e2 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e2]
x0_1e3 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e3]
x0_1e4 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e4]
x0s = [x0_1e2, x0_1e3, x0_1e4]

toleranceGrad = 1e-5
toleranceChange = 1e-9
xConvTol = 1e-6
lineSearchFn = "strong_wolfe"
maxIter = 1000

def closure():
    optimizer.zero_grad()
    curEval = evalFunc(x=x[0])
    curEval.backward(retain_graph=True)
    return curEval

for x0 in x0s:
    ptTrueMin = torch.ones(len(x0))
    jnpTrueMin = jnp.ones(len(x0))
    xPT = torch.tensor(copy.deepcopy(x0), dtype=torch.double)
    xPT.requires_grad = True
    x = [xPT]
    optimizer = torch.optim.LBFGS(x, max_iter=maxIter, line_search_fn=lineSearchFn, tolerance_grad=toleranceGrad, tolerance_change=toleranceChange)
    tStart = time.time()
    optimizer.step(closure)
    elapsedTime = time.time()-tStart
    statePT = optimizer.state[optimizer._params[0]]
    funcEvalsPT = statePT["func_evals"]
    nIterPT = statePT["n_iter"]
    print("x0={}\n".format(x0))
    print("\tPytorch")
    print("\t\tConverged: {}".format(torch.norm(xPT-ptTrueMin, p=2)<xConvTol))
    print("\t\tFunction evaluations: {:d}".format(funcEvalsPT))
    print("\t\tIterations: {:d}".format(nIterPT))
    print("\t\tElapsedTime: {}\n".format(elapsedTime))

    minimizeOptions = {'gtol': toleranceGrad, 'maxiter': maxIter}
    jx0 = jnp.array(x0)
    tStart = time.time()
    optimRes = jax.scipy.optimize.minimize(fun=jEvalFunc, x0=jx0, method='BFGS', options=minimizeOptions)
    elapsedTime = time.time()-tStart

    print("\tjax.scipy.optimize")
    print("\t\tConverged: {}".format(jnp.linalg.norm(optimRes.x-jnpTrueMin, ord=2)<xConvTol))
    print("\t\tFunction evaluations: {:d}".format(optimRes.nfev))
    print("\t\tIterations: {:d}".format(optimRes.nit))
    print("\t\tElapsedTime: {}\n\n".format(elapsedTime))