from typing import List, Tuple, Union, overload

# ArrayLike = Union[float, List[float], Tuple[float]]
import numpy.typing as npt


def a(x: npt.ArrayLike) -> float:
    return 1.2


def z(x: str) -> str:
    return x


b = a([1, 2.1, 3])
b = a("sfasdf")

b = a((1, 2, 3))


b = z(12)
b = z("sdfsdfs")

# -----


@overload
def s(x: str, z: bool = False) -> bool:
    ...


@overload
def s(x: int, z: bool = False) -> bool:
    ...


def s(x, z=False) -> bool:
    return z


print(s("foo"))
print(s(12))
