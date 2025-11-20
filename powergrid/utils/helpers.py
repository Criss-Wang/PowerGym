"""General utility helper functions."""

import uuid


def gen_uuid() -> str:
    """Generate a unique identifier string.

    Returns:
        UUID string (e.g., '550e8400-e29b-41d4-a716-446655440000')

    Examples:
        >>> uid = gen_uuid()
        >>> len(uid)
        36
        >>> '-' in uid
        True
    """
    return str(uuid.uuid4())
