import torch

# NB: We load libtorch.so with RTLD_GLOBAL for UBSAN, unlike our
      # default behavior.
      #
      # The reason for this is that without RTLD_GLOBAL, if we load multiple
      # libraries that depend on libtorch (as is the case with C++ extensions), we
      # will get multiple copies of libtorch in our address space.  When UBSAN is
      # turned on, it will do a bunch of virtual pointer consistency checks which
      # won't work correctly.  When this happens, you get a violation like:
      #
      #    member call on address XXXXXX which does not point to an object of
      #    type 'std::_Sp_counted_base<__gnu_cxx::_Lock_policy::_S_atomic>'
      #    XXXXXX note: object is of type
      #    'std::_Sp_counted_ptr<torch::nn::LinearImpl*, (__gnu_cxx::_Lock_policy)2>'
      #
      # (NB: the textual types of the objects here are misleading, because
      # they actually line up; it just so happens that there's two copies
      # of the type info floating around in the address space, so they
      # don't pointer compare equal.  See also
      #   https://github.com/google/sanitizers/issues/1175
      #
      # UBSAN is kind of right here: if we relied on RTTI across C++ extension
      # modules they would indeed do the wrong thing;  but in our codebase, we
      # don't use RTTI (because it doesn't work in mobile).  To appease
      # UBSAN, however, it's better if we ensure all the copies agree!
      #
      # By the way, an earlier version of this code attempted to load
      # libtorch_python.so with LD_PRELOAD, which has a similar effect of causing
      # it to be loaded globally.  This isn't really a good idea though, because
      # it depends on a ton of dynamic libraries that most programs aren't gonna
      # have, and it applies to child processes.