from typing import overload
from functools import partial

# class tmp:

#     @overload
#     def __init__(self) -> None:
#         ...

#     @overload
#     def __init__(self, a: None, b: None) -> None:
#         ...

#     @overload
#     def __init__(self, a: int, b: int) -> None:
#         ...

#     def __init__(self, a, b) -> None:
#         self.a = a
#         self.b = b


def func(input, p1=None, p2=None):
    print(input, p1, p2)

presetfunc = partial(func)

presetfunc(5)

# if __name__ == "__main__":

#     # t = tmp()

#     # print(t.a)

l1 = [1, 2, 3, 4]
l2 = [2, 3, 4, 5]

for i, (a, b) in enumerate(zip(l1, l2)):
    print(i, a, b)