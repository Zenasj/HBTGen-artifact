import torch

# Never replace unbacked symbols with other unbacked symbols.
                # This is error prone because you can cause references to
                # unbacked symbols to time travel backwards.  E.g.,
                #
                # u1 = x.item()
                # ... use of u1 ...
                # u2 = y.item()
                # u3 = z.item()
                # torch._check(u1 == u2 + u3)
                #
                # If you replace u1 with u2 + u3, then the use of u1 now
                # references u2 and u3 prior to them actually being bound at
                # runtime.  It's pretty inconvenient to setup control
                # dependencies for substitutions, so ban it entirely.