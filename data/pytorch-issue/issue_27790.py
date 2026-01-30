a = [1, 2, 3]
for val in a:
   last = a
print(last) # error: undefined variable last

a = (1, 2, 3)
for val in a:
    last = a
print(last) # no error