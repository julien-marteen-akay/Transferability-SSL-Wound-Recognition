def are_in(objects, target):
    """Checks if all objects are in target"""
    if not is_instance_of(objects):
        raise ValueError("objects must be either a list, tuple or set of the elements in question.")

    for obj in objects:
        if obj not in target:
            return False
    return True


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_instance_of(obj, targets=(list, tuple, set)):
    for target in targets:
        if isinstance(obj, target):
            return True
    return False


def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
    

def is_even(num: int):
    return (abs(num) + 2) % 2 == 0


def is_float(string):
    if string.replace(".", "").isnumeric():
        return True
    else:
        return False
    