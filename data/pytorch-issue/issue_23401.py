import sys, daemon, torch as th
print('BEFORE:', th.cuda.device_count())
with daemon.DaemonContext(stdout=sys.stdout, stderr=sys.stderr, chroot_directory=None, working_directory='.'):
    print('prev_idx:', th._C._cuda_getDevice())

import sys, daemon, torch as th
with daemon.DaemonContext(stdout=sys.stdout, stderr=sys.stderr, chroot_directory=None, working_directory='.'):
    print('AFTER:', th.cuda.device_count())
    print('prev_idx:', th._C._cuda_getDevice())

import multiprocessing.util
import multiprocessing as mp

def _hello(n):
    print("Hello from {}".format(n))
    sys.exit(0)

def _after_fork(arg):
    print("After fork called successfully")
mp.util.register_after_fork(_after_fork, _after_fork)

import os, sys
if os.fork() == 0:
    _hello("fork")

p = mp.Process(target=_hello, args=("mp.Process",))
p.start(); p.join()