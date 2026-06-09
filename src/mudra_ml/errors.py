"""Errors MudraML raises on purpose for data it cannot handle.

Every MudraError carries a clear message that names the offending column,
states the problem, and suggests a fix, so a user never sees a raw pandas or
scikit-learn traceback.
"""

from __future__ import annotations


class MudraError(Exception):
    """Base class for every error MudraML raises on purpose."""


class DataError(MudraError):
    """The data cannot be modelled as asked.

    Raised for conditions the library genuinely cannot handle, such as a
    single-class target or a class with too few examples to split and
    cross-validate.
    """
