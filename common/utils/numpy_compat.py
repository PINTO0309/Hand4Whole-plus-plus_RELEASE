import numpy as np


def patch_numpy_legacy_aliases():
    aliases = {
        'bool': bool,
        'int': int,
        'float': float,
        'complex': complex,
        'object': object,
        'unicode': str,
        'str': str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
