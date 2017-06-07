from sys import _getframe


def funcname():
    return _getframe(1).f_code.co_name
