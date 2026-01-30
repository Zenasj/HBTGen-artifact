x = ExternKernel.require_contiguous(x)
storage, old_layout = as_storage_and_layout(x, want_contiguous=True)

x = ExternKernel.realize_input(x)
storage, old_layout = as_storage_and_layout(x, want_contiguous=True)

# making the base of x contiguous or stride_ordered will not necessarily make
        # the ReinterpretView either, so don't pass along those arguments