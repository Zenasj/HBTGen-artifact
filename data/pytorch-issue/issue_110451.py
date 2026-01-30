# NB, the return handle here represents a temporary tensor, which will be automatically
            # released.
            # Here's a sample usage in the cpp wrapper code:
            #

# RAIIAtenTensorHandle(tmp_tensor_handle_0) will be released after the call to addmm_out.
            # This could be problematic when it's used in a different pattern, for example:
            #