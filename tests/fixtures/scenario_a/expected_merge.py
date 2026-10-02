"""Data processing module."""

def process(data, strict=False):
    """Process a list of numeric data and return doubled values.

    Args:
        data: Input data list.
        strict: If True, raise TypeError for non-numeric values
                instead of skipping them.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data).__name__}")
    result = []
    for item in data:
        if strict and not isinstance(item, (int, float)):
            raise TypeError(f"Non-numeric value: {item!r}")
        if isinstance(item, (int, float)):
            result.append(item * 2)
    return result


def summarize(data, strict=False):
    """Return summary statistics for processed data."""
    processed = process(data, strict=strict)
    if not processed:
        return {
            "count": 0,
            "total": 0,
            "average": 0,
        }
    return {
        "count": len(processed),
        "total": sum(processed),
        "average": sum(processed) / len(processed),
    }
