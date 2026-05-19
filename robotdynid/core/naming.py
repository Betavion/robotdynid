"""Stable public naming helpers."""


def base_parameter_name(index: int) -> str:
    """Return a stable 1-based base-parameter identifier."""
    if index < 1:
        raise ValueError("Base-parameter indices must be 1-based positive integers.")
    return f"bip{index:02d}"

