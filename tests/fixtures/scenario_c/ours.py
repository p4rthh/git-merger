"""Range validator module."""


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_range(value: int, min_val: int, max_val: int) -> bool:
    """Validate that value is within [min_val, max_val] — INCLUSIVE upper bound.

    Returns True if valid, raises ValidationError if not.
    """
    if value < min_val:
        raise ValidationError(f"{value} is below minimum {min_val}")
    if value > max_val:
        raise ValidationError(f"{value} is above maximum {max_val}")
    return True


def validate_batch(values: list[int], min_val: int, max_val: int) -> dict:
    """Validate a batch of values. Returns summary of results."""
    results = {"valid": [], "invalid": []}
    for v in values:
        try:
            validate_range(v, min_val, max_val)
            results["valid"].append(v)
        except ValidationError:
            results["invalid"].append(v)
    return results
