def fn(x, y, b, a):
    ...

# The following calls should return the same result
fn(x, y, b, a)
fn(x, y, b=b, a=a)
fn(x, y, a=a, b=b)
fn(x, y=y, a=a, b=b)
fn(x, a=a, b=b, y=y)

def fn(x, y):
    ...

aot_fn = aot_function(fn, ...)

x, y = ..., ...
res1 = fn(x=x.clone(), y=y.clone())
res2 = fn(y=y.clone(), x=x.clone())
# res1 == res2

aot_res1 = aot_fn(x=x.clone(), y=y.clone())
aot_res2 = aot_fn(y=y.clone(), x=x.clone())
# res1 == aot_res1
# aot_res1 != aot_res2