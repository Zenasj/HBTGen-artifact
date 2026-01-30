import torch

with torch.autograd.profiler.profile(with_stack=True) as prof:
    x = torch.randn((1, 1), requires_grad=True)
    for _ in range(100):
        y = x ** x
        y.backward()
prof.export_chrome_trace('trace.json')

def export_chrome_trace(self, path):
        self._check_finish()
        # if kineto_available():
        #     self.kineto_results.save(path)  # type: ignore[union-attr]
        # else:
        return self.function_events.export_chrome_trace(path)  # type: ignore[union-attr]