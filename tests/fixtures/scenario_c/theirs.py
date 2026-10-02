"""Range validator module."""


class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, message: str, code: str, value: int):
        super().__init__(message)
        self.code = code
        self.value = value


def validate_range(value: int, min_val: int, max_val: int) -> bool:
    """Validate that value is within [min_val, max_val) — exclusive upper bound.

    Returns True if valid, raises ValidationError with error code if not.
    """
    if value < min_val:
        raise ValidationError(
            f"{value} is below minimum {min_val}",
            code="BELOW_MIN",
            value=value,
        )
    if value >= max_val:
        raise ValidationError(
            f"{value} is at or above maximum {max_val}",
            code="ABOVE_MAX",
            value=value,
        )
    return True


def validate_batch(values: list[int], min_val: int, max_val: int) -> dict:
    """Validate a batch of values. Returns summary with error details."""
    results = {"valid": [], "invalid": [], "errors": []}
    for v in values:
        try:
            validate_range(v, min_val, max_val)
            results["valid"].append(v)
        except ValidationError as e:
            results["invalid"].append(v)
            results["errors"].append({
                "value": e.value,
                "code": e.code,
                "message": str(e),
            })
    return results
