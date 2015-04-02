
from numpy import ndarray, zeros
from openmdao.main.api import VariableTree


def init_vartree(vt, ni):
    """set VariableTree array variables recursively to size ni"""

    for name in vt.list_vars():
        var = getattr(vt, name)

        if isinstance(var, VariableTree):
            var = init_vartree(var, ni)

        elif isinstance(var, ndarray):
            setattr(vt, name, zeros(ni))

    return vt