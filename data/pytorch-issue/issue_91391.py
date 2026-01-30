if not is_forward_ad and any(o.is_complex() for o in outputs):
        raise ValueError("Expected output to be non-complex. get_numerical_jacobian no "
                         "longer supports functions that return complex outputs.")