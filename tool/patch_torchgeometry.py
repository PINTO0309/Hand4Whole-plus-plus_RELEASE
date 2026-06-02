"""Patch torchgeometry for modern PyTorch bool mask behavior."""

from __future__ import annotations

import inspect
from pathlib import Path

import torchgeometry.core.conversions as conversions


OLD = """    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
"""

NEW = """    mask_c0 = mask_d2.float() * mask_d0_d1.float()
    mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())
    mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()
    mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())
"""


def main() -> None:
    path = Path(inspect.getfile(conversions))
    text = path.read_text()

    if NEW in text:
        print(f"Already patched: {path}")
        return
    if OLD not in text:
        raise SystemExit(f"Expected torchgeometry mask block was not found: {path}")

    path.write_text(text.replace(OLD, NEW, 1))
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
