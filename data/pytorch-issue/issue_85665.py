# b * c is computed twice here
a = b * c + g;
d = b * c * e;

# After CSE, the same code is rewritten as
tmp = b * c;
a = tmp + g;
d = tmp * e;

# b * c now can be reused in other places, such as `a` and `d`