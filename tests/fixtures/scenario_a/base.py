"""Data processing module."""

def process(data):
    """Process a list of numeric data and return doubled values."""
    result = []
    for item in data:
        result.append(item * 2)
    return result


def summarize(data):
    """Return summary statistics for processed data."""
    processed = process(data)
    return {
        "count": len(processed),
        "total": sum(processed),
        "average": sum(processed) / len(processed),
    }
