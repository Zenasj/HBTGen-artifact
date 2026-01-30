if dst.data_ptr() != src.data_ptr():
    dst.data._copy(src)