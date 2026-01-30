my_fn = foo.bar.my_fn(input)
my_fn_1 = foo.bar.my_fn(input2)

my_fn = my_fn(input)  # shadowed name!
my_fn_1 = my_fn(input)  # uh oh