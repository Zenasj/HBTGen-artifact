time_start = time.time()
for i in range(1000):
    self.check_graph(MyModule(), inputs)
warnings.warn(f"time_taken: {time.time() - time_start}")