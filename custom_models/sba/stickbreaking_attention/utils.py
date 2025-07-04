import torch
from typing import Callable, Iterable, Sequence
import math

PACKAGE_NAME = "stickbreaking_attention"
log2 = math.log(2)
inv_log2 = 1 / log2
ALLOW_TF32 = True



def _dispatch(func: Callable, compileable_fn: Callable, *args, **kwargs):
    if torch.compiler.is_compiling():
        output = compileable_fn(*args, **kwargs)
    else:
        output = func(*args, **kwargs)
    return output


def custom_op(
    name: str = None,
    mutates_args: str | Iterable[str] = None,
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
) -> Callable:
    compileable_name = f"{PACKAGE_NAME}::{name}"

    def _inner(func: Callable):
        compileable_func = torch.library.custom_op(
            compileable_name, func, mutates_args=mutates_args, device_types=device_types, schema=schema
        )

        def _run(*args, **kwargs):
            return _dispatch(func, compileable_func, *args, **kwargs)
        # _run.__signature__ = inspect.signature(func)
        # _run.__name__ = func.__name__
        return _run

    return _inner
