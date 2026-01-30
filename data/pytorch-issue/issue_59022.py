import torch
from textwrap import dedent

src = dedent(
                """\
                fake_name = "time"
                __import__(fake_name) # need to throw error

                def foo(mod_name: str):
                    __import__(mod_name) # need to not throw error? this PR will flag this as an error .-.
                """
            )

with torch.package.PackageExporter("output", verbose=False) as e:
    e.save_source_string('broken',  src)