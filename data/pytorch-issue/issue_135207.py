import torch

def post_acc_grad_hook(self, input, hook_id):
    assert isinstance(input, torch.Tensor)
    assert self.hooks_proxy is not None
    hook = self.hooks_proxy[hook_id]  # type: ignore[index]
    proxies = self.proxy_call_hook(hook, input) 
    with disable_proxy_modes_tracing():
        input = [maybe_clone(input)]
        self.bind_tensors_to_proxies(input, proxies)
        
    return input

def bind_tensors_to_proxies(self, tensors, proxies):
    if isinstance(proxies, torch.fx.Proxy):
        proxies = [proxies[i] for i in range(len(tensors))]
    assert len(tensors) == len(proxies)
    track_tensor_tree(tensors, proxies, constant=None, tracer=self.fx_tracer)

proxies = [proxies[i] for i in range(len(tensors))]

def post_acc_grad_hook(self, input, hook_id):
        assert isinstance(input, torch.Tensor)
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        # was proxies = self.proxy_call_hook(hook, input) 
        proxy = self.proxy_call_hook(
            hook,
            input,
        )
        with disable_proxy_modes_tracing():
            input = [maybe_clone(input)]
            # was  self.bind_tensors_to_proxies(input, proxies)
            self.bind_tensors_to_proxies(input, [proxy])
        return input

def tensor_pre_hook(self, inputs, hook_id, i: int):
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxy = self.proxy_call_hook(
            hook,
            inputs[i],
        )
        with disable_proxy_modes_tracing():
            inputs[i] = maybe_clone(inputs[i])
            self.bind_tensors_to_proxies([inputs[i]], [proxy])
        return inputs