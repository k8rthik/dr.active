"""Target-name resolution and the stable target ordering used by features."""

from __future__ import annotations

from .config import TARGET_ALIASES, TARGET_NAMES, TARGETS, Target


class UnknownTargetError(ValueError):
    """Raised when a user-supplied target name is not one we model."""


_BY_NAME: dict[str, Target] = {t.name: t for t in TARGETS}
_BY_CHEMBL_ID: dict[str, Target] = {t.chembl_id.upper(): t for t in TARGETS}
_INDEX: dict[str, int] = {name: i for i, name in enumerate(TARGET_NAMES)}


def known_target_names() -> tuple[str, ...]:
    return TARGET_NAMES


def describe_targets() -> tuple[Target, ...]:
    return TARGETS


def _unknown(raw: object) -> UnknownTargetError:
    return UnknownTargetError(
        f"Unknown target {raw!r}. Known targets: {', '.join(TARGET_NAMES)}."
    )


def resolve_target(raw: str) -> Target:
    """Resolve a user string (name, alias or ChEMBL id) to a Target."""
    if not isinstance(raw, str):
        raise _unknown(raw)
    key = raw.strip().upper()
    if not key:
        raise _unknown(raw)
    name = TARGET_ALIASES.get(key)
    if name is None:
        raise _unknown(raw)
    return _BY_NAME[name]


def resolve_target_name(raw: str) -> str:
    return resolve_target(raw).name


def target_index(raw: str) -> int:
    """Position of a target in the canonical ordering (feature one-hot slot)."""
    return _INDEX[resolve_target(raw).name]


def target_from_chembl_id(chembl_id: str) -> Target | None:
    """Map a ChEMBL target id to a Target, or None if we do not model it."""
    if not isinstance(chembl_id, str):
        return None
    return _BY_CHEMBL_ID.get(chembl_id.strip().upper())
