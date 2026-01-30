def expect_true(self, file, line):
        if self.has_hint():
            # OK to generate guards
            return self.guard_bool(file, line)
        # Generate a deferred runtime assert (this might actually end up doing
        # a regular guard if we can!)
        # TODO: file/line here is very important, because the assert has been
        # deferred so you can't backtrace easily
        return self.shape_env.defer_runtime_assert(
            self.expr, f"{file}:{line}", fx_node=self.fx_node
        )