mod.sub1 = FSDP(mod.sub1)
mod.sub2 = FSDP(mod.sub2)
mod = FSDP(mod)

mod = FSDP(mod, auto_wrap_policy=ModuleWrapPolicy({Submodule}))

fully_shard(mod.sub1)
fully_shard(mod.sub2)
fully_shard(mod)

fully_shard(mod, policy=ModuleWrapPolicy({Submodule}))