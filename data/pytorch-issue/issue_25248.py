matrices = st.one_of([
    tensors(shape, dtype, elements=elems)
    for elems in (st.floats(), st.floats(-10, 10), st.floats(-2, 2))  # or combinations of such
])