import mymodule
print(mymodule.inc())  # 1
print(mymodule.inc())  # 2
import sys
del sys.modules['mymodule']
import mymodule
print(mymodule.inc())  #1