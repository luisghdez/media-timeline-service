"""Stack two videos onto a 9:16 canvas (top 1/4 + bottom 3/4)."""

from .compose import compose
from .schema import ComposeResult

__all__ = ["ComposeResult", "compose"]
