import atexit
import json
import sys
from typing import Callable, Dict

Point = tuple[int, int]


def is_in_script_args(arg: str) -> bool:
    """Checks if an argument is contained in the script arguments."""
    return any(arg in script_arg for script_arg in sys.argv[1:])


def file_cache(
    file_name: str,
) -> Callable[[Callable[[str], int]], Callable[[str], int]]:
    """Decorator to cache the results of a function to a file"""
    try:
        cache: Dict[str, int] = json.load(open(file_name, "r"))
    except (IOError, ValueError):
        cache = {}

    atexit.register(lambda: json.dump(cache, open(file_name, "w"), indent=4))

    def decorator(func: Callable[[str], int]) -> Callable[[str], int]:
        def new_func(param: str) -> int:
            if param not in cache:
                cache[param] = func(param)
            return cache[param]

        return new_func

    return decorator
