import torch

from junitparser import JUnitXml
import re
from collections import defaultdict
from pprint import pprint

xml = JUnitXml.fromfile('foo.xml')

result = defaultdict(int)

for suite in xml:
    for case in suite:
        if case.result:
            r = case.result[0]
            if r.type == "pytest.skip" and r.message.startswith("not implemented"):
                m = re.search(r"Could not run '([^']+)'", r.text)
                if m:
                    op = m.group(1)
                elif "torch.tensor" in r.text:
                    op = "torch.tensor"
                else:
                    continue
                result[op] += 1

pprint(result)