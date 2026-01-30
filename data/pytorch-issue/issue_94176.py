def RETURN_VALUE(self, inst):
        if self.output.count_calls() == 0:
            raise exc.SkipFrame()  # here
        self.output.compile_subgraph(self) # and here