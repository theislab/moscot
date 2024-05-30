from itertools import combinations


def _check_is_copy(o1: object, o2: object, shallow_copy: tuple[str, ...]) -> bool:
    if type(o1) is not type(o2):
        return False

    for k in o1.__dict__:
        v1 = getattr(o1, k)
        v2 = getattr(o2, k)
        if type(v1) is not type(v2):
            return False
        # these basic types are treated differently in python and there's no point in comparing their ids
        if isinstance(v1, (str, int, bool, float)) or v1 is None:
            continue
        if k in shallow_copy:
            if id(v1) != id(v2):
                return False
        else:
            if id(v1) == id(v2):
                return False

    return True


def check_is_copy_multiple(os: tuple[object, ...], shallow_copy: tuple[str, ...]) -> bool:
    if len(os) < 1:
        return False
    combs = combinations(os, 2)
    return all(_check_is_copy(o1, o2, shallow_copy) for o1, o2 in combs)
