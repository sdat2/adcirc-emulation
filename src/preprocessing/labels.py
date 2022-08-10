"""Sanitize labels etc."""


def sanitize(input: str) -> str:
    output = input.replace("_", " ")
    return output.title()
