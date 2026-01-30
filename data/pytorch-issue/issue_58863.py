import torch

with torch.package.PackageExporter("output") as e:
    e.save_source_string('m',  '__import__("these", dont, have, to, be, contants)')

import torch
from textwrap import dedent

src = dedent(
                """\
                fake_name = "time"
                __import__(fake_name) # need to throw error because time won't be in package

                def foo(mod_name: str):
                    __import__(mod_name) # need to not throw error because isn't a real dependency
                """
            )

with torch.package.PackageExporter("output", verbose=False) as e:
    e.save_source_string('m',  src)


importer = torch.package.PackageImporter("output")
importer.import_module("m") # will fail to be able to resolve module 'time'

from pkg_resources import declare_namespace as _declare_namespace
_declare_namespace(__name__)