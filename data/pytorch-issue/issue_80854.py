def fn(a, b):
    return a.relu_() + b.relu()
def fn(a, b):
    c = a + b
    return c.sigmoid().add(c.relu_())