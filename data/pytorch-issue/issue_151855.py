import torch
import torch.nn as nn

def test_runtime_checks_error_msg(self):

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return a + b

            torch.library.impl_abstract("mylib::foo", lib=lib)
            def foo_fake_impl(a, b):
                return a + b


            class Model(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                def forward(self, x):
                    for i in range(10000):
                        x = torch.ops.mylib.foo(x, x)
                    return x

            inputs = (torch.ones(8, 8, 8), )
            model = Model()
            with self.assertRaisesRegex(Exception, "torch._inductor.config.aot_inductor.compile_wrapper_opt_level"):
                with torch.no_grad():
                    AOTIRunnerUtil.compile(
                        model,
                        inputs,
                    )