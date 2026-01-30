set_logs(aot=logging.DEBUG)
set_logs(dynamo=logging.DEBUG)

import torch
import logging

inductor_schedule_log = torch._logging.getArtifactLogger('torch._inductor', 'schedule')

def example_inductor_func():
    if inductor_schedule_log.isEnabledFor(logging.DEBUG):
        inductor_schedule_log.debug('message')

print('----------------------')
torch._logging.set_logs(inductor=logging.DEBUG, schedule=False)
example_inductor_func()

print('----------------------')
torch._logging.set_logs(inductor=logging.DEBUG, schedule=True)
example_inductor_func()

print('----------------------')
torch._logging.set_logs(inductor=logging.CRITICAL, schedule=True)
example_inductor_func()


aot_schedule_log = torch._logging.getArtifactLogger('torch._functorch.aot_autograd', 'schedule')

def example_aot_func():
    if aot_schedule_log.isEnabledFor(logging.DEBUG):
        aot_schedule_log.debug('message')

print('----------------------')
torch._logging.set_logs(aot=logging.DEBUG, schedule=False)
example_aot_func()

print('----------------------')
torch._logging.set_logs(aot=logging.DEBUG, schedule=True)
example_aot_func()

print('----------------------')
torch._logging.set_logs(aot=logging.CRITICAL, schedule=True)
example_aot_func()