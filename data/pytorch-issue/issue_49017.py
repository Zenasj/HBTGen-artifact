def gen():
    for i in range(10):
        yield i
    
    # return with a value is automatically converted to StopIteration
    return 10

for i in gen():
    print(i)

it = gen()
try:
    while True:
        print(next(it))

except StopIteration as e:
    print(type(e), e, e.value)

def gen():
    for i in range(10):
        yield i

it = gen()
try:
    while True:
        print(next(it))

except StopIteration as e:
    print(type(e), e, e.value)

# ctx.py
import torch

def gen():
    with torch.no_grad():
        yield 1  # some compute, e.g. torch.matmul(...)
    yield 2

with torch.enable_grad():
    it = gen()
    print(torch.is_grad_enabled())
    print(next(it))
    print(torch.is_grad_enabled())
    print(next(it))
    print(torch.is_grad_enabled())

class ctx:
    def __enter__(self):
        print('>>> enter')
    def __exit__(self, typ, val, tb):
        print('<<< exit')

def gen():
    with ctx():
        yield 1
    yield 2

it = gen()
print(next(it))
print('do something')
print(next(it))

# ctx.py
import torch

def gen():
    print("inside 0 ", torch.is_grad_enabled())
    yield 1
    print("inside 1 ", torch.is_grad_enabled())
    yield 2
    print("inside 2 ", torch.is_grad_enabled())

def no_grad_wrapper(gen):
    def wrapped():
        g = gen()
        with torch.no_grad():
            resp = g.send(None)

        while True:
            # Simplified version of the wrapper in this PR that
            # wraps the send but not the yield
            req = yield resp

            with torch.no_grad():
                resp = g.send(req)

    return wrapped

print("default:")
with torch.enable_grad():
    it = gen()
    print(torch.is_grad_enabled())
    print(next(it))
    print(torch.is_grad_enabled())
    print(next(it))
    print(torch.is_grad_enabled())

print("")
print("wrapped:")
with torch.enable_grad():
    it = no_grad_wrapper(gen)()
    print(torch.is_grad_enabled())
    print(next(it))
    print(torch.is_grad_enabled())
    print(next(it))
    print(torch.is_grad_enabled())