# Enables a local, filesystem "profile" which can be used for automatic
# dynamic decisions, analogous to profile-guided optimization.  The idea is
# that if we observe that a particular input is dynamic over multiple
# iterations on one run, we can save a profile with this information so the
# next time we run we can just make it dynamic the first time around, skipping
# an unnecessary static compilation.  The profile can be soundly stale, if it
# is wrong, it just means we may make more things dynamic than was actually
# necessary (NB: this /can/ cause a failure if making something dynamic causes
# the compiler to stop working because you tickled a latent bug.)
#
# The profile is ONLY guaranteed to work if the user source code is 100%
# unchanged.  Applying the profile if there are user code changes is only
# best effort otherwise.  In particular, we identify particular code objects
# by filename, line number and name of their function, so adding/removing newlines
# will typically cause cache misses.  Once a profile is created, it will
# never be subsequently updated, even if we discover on a subsequent run that
# more inputs are dynamic (TODO: add some way of manually clearing the
# profile in a convenient way; TODO: add a way of not doing this behavior).
#
# Enabling this option can potentially change the automatic dynamic behavior
# of your program, even when there is no profile.  Specifically, we uniquely
# identify a code object by its filename/line number/name.  This means if you
# have multiple distinct code objects that have identical filename/line
# number, we will share automatic dynamic information across them (TODO:
# change default automatic dynamic behavior so it also crosstalks in this way)
automatic_dynamic_local_pgo = False