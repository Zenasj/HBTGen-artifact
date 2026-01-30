import torch as T
import os

freqs = T.cuda.FloatTensor([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.03178183361887932, 0.027680952101945877, 0.033176131546497345,
    0.046052902936935425, 0.07742464542388916, 0.11543981730937958,
    0.14148041605949402, 0.15784293413162231, 0.13180233538150787,
    0.08271478116512299, 0.049702685326337814, 0.027557924389839172,
    0.018125897273421288, 0.011851548217236996, 0.010252203792333603,
    0.007422595750540495, 0.005372154992073774, 0.0045109698548913,
    0.0036087757907807827, 0.0035267581697553396, 0.0018864056328311563,
    0.0024605290964245796, 0.0022964938543736935, 0.0018453967059031129,
    0.0010662291897460818, 0.0009842115687206388, 0.00045109697384759784,
    0.0007791675161570311, 0.00020504408166743815, 0.00020504408166743815,
    0.00020504408166743815, 0.00012302644609007984, 0.0,
    0.00012302644609007984, 4.100881778867915e-05, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0])

for i in range(1_000_000_000):
    state = T.cuda.get_rng_state()
    sample = T.multinomial(freqs, 1000, True)
    if freqs[sample].min() == 0:
        break

print(72*'-')
print(f'failure after {i} iterations')
sample_idx = (freqs[sample] == 0).nonzero()[0][0]
sampled = sample[sample_idx]
print(f'{sample_idx}th element of last sample was {sampled}, '
      f'which has probability {freqs[sampled]}')

print(72*'-')
SAVE_PATH = 'failed_state.pt'
T.save(state, open(SAVE_PATH, 'wb'))
print(f'\nRNG state saved to `{SAVE_PATH}`')

T.cuda.set_rng_state(T.load(open(SAVE_PATH, 'rb')))
assert freqs[T.multinomial(freqs, 1000, True)].min() == 0

print(72*'-')
print(f'\npytorch version: {T.__version__}\n')
print(72*'-')
print('Operating system:')
os.system('lsb_release -a')
print(72*'-')
print('\nGPU information:')
os.system('nvidia-smi')
print(72*'-')
print('\nCUDA library information:')
os.system('dpkg -l | grep -i cuda')