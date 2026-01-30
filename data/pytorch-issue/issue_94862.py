m1 = Module1()
m2 = Module2()
def f(inp):
    return m2(m1(inp))
gm, _ = dynamo.export(f, inp)