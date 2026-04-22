# -*- coding: utf-8 -*-
"""
Test harness setup.

`flame.data` imports `torchtitan.tools.logging.logger` and `torchtitan.tools.utils.Color`.
In CI we don't want to install the full torchtitan package (heavy, and its pip version
has drifted from the version flame targets). Stub out just the two symbols we need
when torchtitan isn't importable so `import flame.data` works in a lightweight env.
"""
from __future__ import annotations

import logging
import sys
import types


def _install_torchtitan_stub() -> None:
    try:
        import torchtitan.tools.logging  # noqa: F401
        import torchtitan.tools.utils  # noqa: F401
        return
    except Exception:
        pass

    tt = sys.modules.setdefault("torchtitan", types.ModuleType("torchtitan"))
    tt_tools = sys.modules.setdefault("torchtitan.tools", types.ModuleType("torchtitan.tools"))
    tt_tools_logging = sys.modules.setdefault(
        "torchtitan.tools.logging", types.ModuleType("torchtitan.tools.logging")
    )
    tt_tools_utils = sys.modules.setdefault(
        "torchtitan.tools.utils", types.ModuleType("torchtitan.tools.utils")
    )

    tt_tools_logging.logger = logging.getLogger("flame.test")

    class _Color:
        black = red = green = yellow = blue = magenta = cyan = white = ""
        reset = ""
    tt_tools_utils.Color = _Color

    tt.tools = tt_tools
    tt_tools.logging = tt_tools_logging
    tt_tools.utils = tt_tools_utils


_install_torchtitan_stub()
