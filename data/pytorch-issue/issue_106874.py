def init_backend_registration(self):
    if get_scheduling_for_device("cpu") is None:
        from .codegen.cpp import CppScheduling

        register_backend_for_device("cpu", CppScheduling, WrapperCodeGen)

    if get_scheduling_for_device("cuda") is None:
        from .codegen.triton import TritonScheduling

        register_backend_for_device("cuda", TritonScheduling, WrapperCodeGen)