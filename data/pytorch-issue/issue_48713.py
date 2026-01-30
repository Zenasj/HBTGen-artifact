import sys
import torch
import functools

class _coroMixin:
    def _wrap_generator(self, func):
        """allow two-way comm with the enhanced generators from pep-342"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)

            # the logic resembles pep-380 `yield from`, which we cannot use
            # because the wrapped generator must be interacted with from within
            # the grad-mode context manager. At the same time our own yields
            # must take place outside of the grad-mode context.
            try:
                # wait until .send(None) from the caller, then fire up the wrapped
                # generator by issuing our own `None` to it.
                with self:
                    response = gen.send(None)

                while True:
                    try:
                        # Relay the `response` to the caller and get its next request,
                        # whatever it may be. In the common use-cases, e.g. `for` loop
                        # or `next`, the request is always `None`.
                        request = yield response

                    except GeneratorExit:
                        # XXX this section can be merged with BaseException, if we want
                        # this decorator to be completely transparent, e.g. in the case when
                        # the generator yields after close, causing RuntimeError:
                        # with the message `generator ignored GeneratorExit`

                        # close the generator, since our `yield` raised GeneratorExit
                        with self:
                            gen.close()
                        raise

                    except BaseException:
                        # the caller threw an exception at us. Delegate its handling to
                        # the generator. If it raises or stops, then we'll catch it
                        # either way, however if it continues and safely handles the
                        # exception, then we must yield and continue, as if nothing
                        # happened.
                        with self:
                            response = gen.throw(*sys.exc_info())

                    else:
                        # send the recent request to the wrapped generator from within
                        # the grad-mode context, and receive whatever it yielded.
                        with self:
                            response = gen.send(request)

            except StopIteration as e:
                # `return` from a generator raises `StopIteration` with the returned
                # value as a payload in `.args`

                # according to pep-479 _raising_ SI from a generator is a runtime error,
                # but at the same time SI indicates that the generator has terminated
                #"normally". Hence here we catch SI and return its payload, to emulate
                # proper termination via a `return` expression.
                return e.value  # https://docs.python.org/3/library/exceptions.html#StopIteration

        return generator_context

class coro_no_grad(_coroMixin, torch.no_grad):
    pass

@coro_no_grad()
def doubler(x):
    return x * 2

@coro_no_grad()
def multiplier_counter(x, n=10):
    k = 0
    while k < n:
        try:
            val = yield k * x
        
        except FloatingPointError as e:
            print('handled', type(e), str(e))

        if val is None:
            k += 1
        else:
            k = val

x = torch.tensor([1.], requires_grad=True)
z = doubler(x)
assert not z.requires_grad
print(z)

for z in multiplier_counter(x, n=5):
    assert not z.requires_grad
    print(z)

gen = multiplier_counter(x, n=2)

z = next(gen)
assert not z.requires_grad
print(z)

z = next(gen)
assert not z.requires_grad
print(z)

z = next(gen)

gen = multiplier_counter(x, n=3)

# start the generator and continue the current streak
for i in range(3):
    # .send(None): start the generator and get its first
    # yield, or continue the current streak.
    z = gen.send(None)
    assert not z.requires_grad
    print(z)

z = gen.send(-10)  # reset the counter and receive the next value
assert not z.requires_grad
print(z)

for i in range(5):
    z = next(gen)  # can issue `gen.send(None)` too
    assert not z.requires_grad
    print(z)

# throw at the `yield` expression in the generator
print(gen.throw(FloatingPointError('instability')))

# throw a bad value at the generator
print(gen.throw(ValueError('bad value')))

# throws stop iteration, since the generator could not handle the last exception
next(gen)

class multiplier_counter:
    def __init__(self, x, n=10):
        self.x, self.n = x, n
        self.counter = 0

    @torch.no_grad()
    def __next__(self):
        c, self.counter = self.counter, self.counter + 1
        return self.x * c

    def reset(self, value):
        if value is not None:
            self.counter = value