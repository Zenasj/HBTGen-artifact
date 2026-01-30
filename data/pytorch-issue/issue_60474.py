a = "\ "  # This should trigger W605 invalid escape sequence
b = "\\ "  # And this should not
c = r"\ "  # And this should not either, as raw-strings do not need escapes

assert a == b
assert b == c