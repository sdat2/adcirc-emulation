"""Sanitize labels etc."""


def sanitize(input: str) -> str:
    """
    Sanitize the labels.

    Args:
        input (str): input string.

    Returns:
        str: output string.

    Example::
        >>> from src.preprocessing.labels import sanitize
        >>> sanitize("storm_surge")
        'Storm Surge'
    """
    output = input.replace("_", " ")
    return output.title()
