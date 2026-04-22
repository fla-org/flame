# -*- coding: utf-8 -*-

"""flame package root.

This module installs small compatibility shims so that torchtitan >= 0.2
can be imported on top of older torch builds that predate some of the
symbols / inductor options torchtitan assumes. The shims are best-effort
and only suppress import-time failures; they do NOT try to silently
emulate functionality that doesn't exist. If you actually invoke a
feature that relies on a missing primitive (e.g. varlen attention on
torch < 2.10), you get a clear runtime error.
"""

import re
import sys
import types

# ---------------------------------------------------------------------------
# Shim 1: torch.nn.attention.varlen (introduced in torch 2.10). torchtitan
# imports ``varlen_attn`` at module load, so provide a stub if the real
# module is absent.
# ---------------------------------------------------------------------------
if "torch.nn.attention.varlen" not in sys.modules:
    try:
        import torch.nn.attention.varlen  # noqa: F401
    except ImportError:
        import torch.nn.attention as _attn_pkg

        _stub = types.ModuleType("torch.nn.attention.varlen")

        def _missing_varlen_attn(*args, **kwargs):
            raise RuntimeError(
                "torch.nn.attention.varlen.varlen_attn is not available in this "
                "torch build. Upgrade to torch >= 2.10 to use varlen attention."
            )

        _stub.varlen_attn = _missing_varlen_attn
        sys.modules["torch.nn.attention.varlen"] = _stub
        setattr(_attn_pkg, "varlen", _stub)

# ---------------------------------------------------------------------------
# Shim 2: torch.compile option tolerance. torchtitan pins a few inductor
# options (e.g. ``wrap_inductor_compiled_regions``) that only land in newer
# torch builds. On older builds, torch.compile raises RuntimeError at call
# time. Wrap torch.compile so that unknown options are dropped with a
# warning instead of aborting the whole import chain.
# ---------------------------------------------------------------------------
import torch as _torch  # noqa: E402

_orig_compile = _torch.compile
_UNKNOWN_OPT_RE = re.compile(r"Unexpected optimization option (\S+?)[,\s]")


def _tolerant_compile(*args, **kwargs):
    options = kwargs.get("options")
    if not options:
        return _orig_compile(*args, **kwargs)
    fixed = dict(options)
    while True:
        try:
            kwargs["options"] = fixed
            return _orig_compile(*args, **kwargs)
        except RuntimeError as exc:
            m = _UNKNOWN_OPT_RE.search(str(exc))
            if m is None or not fixed:
                raise
            fixed.pop(m.group(1), None)


_tolerant_compile.__wrapped__ = _orig_compile
_torch.compile = _tolerant_compile

__version__ = "0.1.0"
