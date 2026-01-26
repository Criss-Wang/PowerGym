"""General utility helper functions for power domain."""

import uuid


def gen_uuid() -> str:
    """Generate a unique identifier string.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())
