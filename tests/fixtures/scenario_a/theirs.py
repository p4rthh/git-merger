"""Data processing module."""

def process(data):
    """Process a list of numeric data and return doubled values."""
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data).__name__}")
    result = []
    for item in data:
        result.append(item * 2)
    return result


def summarize(data):
    """Return summary statistics for processed data."""
    processed = process(data)
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
