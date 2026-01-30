# Dataclasses is in-built from py >=3.7. This version is a backport for py 3.6.
if (sys.version_info.major, sys.version_info.minor) == (3, 6):
  REQUIRED_PKGS.append('dataclasses')

@dataclasses.dataclass
class CellCopyStats:
  processed_cells: int = 0
  updated_cells: int = 0
  unmatched_target_cells: list[str] = dataclasses.field(default_factory=list)
  unmatched_source_cells: list[str] = dataclasses.field(default_factory=list)
  out_of_order_target_cells: list[str] = dataclasses.field(default_factory=list)

import sys

if sys.version_info < (3, 7):
    REQUIRED_PKGS.append('dataclasses')