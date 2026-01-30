## PRE-PR UX
def f(mode):
  with mode.restore():  # user needs to understand this restore thing?
    ...

with Mode() as m:
  pass
f(m)

## FROM FEEDBACK, USER DESIRED CODE. POSSIBLE POST-PR
def f(mode):
  with mode:
    ...
f(Mode())