from ..utils import misc


def gpu_from_string(s: str):
    # keys which are not in a string list
    if "auto" in s:
        return "auto"
    elif not (s.startswith("[") or s.endswith("]")):
        return int(s)
    # keys which are in a string list
    else:
        devices = to_list(s)
        return [int(idx) for idx in devices]

def boolean_string(s: str):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError("Not a valid boolean string, either 'false' or 'true' (capitalization doesn't matter)\n"
                         f"Your input was: {s}")
    return s == 'true'


def to_list(s: str, strip_whitespace=True):
    """converts a string list, which must start and end with e.g. square brackets like a python list does, to an actual python list"""
    if strip_whitespace:
        s = "".join(s.split())  # e.g. " \t   foo \n bar " --> "foobar"
    return s[1: -1].split(",")
