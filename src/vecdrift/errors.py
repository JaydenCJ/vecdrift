"""Exception hierarchy for vecdrift.

Every error raised on a user-facing path derives from :class:`VecdriftError`
so the CLI can catch one type, print a clean message, and exit with code 2.
"""

from __future__ import annotations


class VecdriftError(Exception):
    """Base class for all vecdrift errors."""


class InputError(VecdriftError):
    """A vector export file is missing, unreadable, or malformed."""


class BaselineError(VecdriftError):
    """A baseline file is malformed or has an unsupported format version."""


class PairingError(VecdriftError):
    """Two anchor sets share too few ids to compare their geometry."""
