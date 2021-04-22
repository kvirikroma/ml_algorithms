def green(inp: str) -> str:
    """Returns string in green color to print"""
    return "\033[92m" + inp + "\033[0m"


def red(inp: str) -> str:
    """Returns string in red color to print"""
    return "\033[91m" + inp + "\033[0m"


def yellow(inp: str) -> str:
    """Returns string in yellow color to print"""
    return "\033[93m" + inp + "\033[0m"


def blue(inp: str) -> str:
    """Returns string in blue color to print"""
    return "\033[94m" + inp + "\033[0m"


def cyan(inp: str) -> str:
    """Returns string in green color to print"""
    return "\033[36m" + inp + "\033[0m"


def magenta(inp: str) -> str:
    """Returns string in red color to print"""
    return "\033[95m" + inp + "\033[0m"


def grey(inp: str) -> str:
    """Returns string in yellow color to print"""
    return "\033[90m" + inp + "\033[0m"
