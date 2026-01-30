import torch

class ExecutableFSDP(FullyShardedDataParallel):
    """
    Adds ``execute()`` to FullyShardedDataParallel, which allows access to unsharded parameters.

    .. note:: This is mostly a copy-paste of https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py#L784-L806 
    """

    # noinspection PyNoneFunctionAssignment
    def execute(self, fcn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Allows user to run the given ``fcn``, which would access unsharded parameters of the model.

        .. note:: Mostly, copy & paste from ``FullyShardedDataParallel.forward()``.
        """
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            unshard_fn = functools.partial(_pre_forward_unshard, self, self._handles)
            reshard_fn = functools.partial(_post_forward_reshard, self, self._handles)
            args, kwargs = _pre_forward(
                self, self._handles, unshard_fn, self._fsdp_wrapped_module, args, kwargs
            )
            for handle in self._handles:
                p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            # !!! this is the only line we change from ``FullyShardedDataParallel.forward()`` !!!
            output = fcn(*args, **kwargs)

            return _post_forward(self, self._handles, reshard_fn, self, unused, output)


class GenerationFSDP(ExecutableFSDP):
    """
    Adds support for ``PreTrainedModel.generate()`` for ``FullyShardedDataParallel`` models.
    """

    def __init__(self, module: PreTrainedModel, *args: Any, **kwargs: Any):
        super().__init__(module, *args, **kwargs)

    def generate(
        self, *args: Any, **kwargs: Any
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Runs ``PreTrainedModel.generate()`` with access to unsharded parameters.
        """

        return self.execute(
            self._fsdp_wrapped_module.generate,
            *args,
            **kwargs,
        )