from typing import overload
from functools import partial


class tmp:
    i = 0

    def __init__(self, a, b) -> None:
        print("class tmp init called.")
        tmp.i += 1
        self.a = a
        self.b = b

    def __repr__(self):
        return "tmp"


class tmp2(tmp):
    def __init__(self, a=1, b=2):
        print("class tmp2 init called.")
        super().__init__(a, b)

    def __repr__(self):
        return "tmp2"


class tmp3(tmp2):
    def __init__(self, a=3, b=3):
        print("class tmp3 init called.")
        super().__init__(a, b)

    def __repr__(self):
        return "tmp3"


if __name__ == "__main__":
    t3 = tmp3()
    t2 = tmp2()

    lst = [t2, t3]

    lst2 = lst[:]
    lst2.append(tmp(1, 2))

    print(t3.a)
    print(t2.a)

    print(isinstance(t3, tmp))
    print(isinstance(t3, tmp2))
    print(isinstance(t2, tmp))
    print(isinstance(t2, tmp2))
    print(isinstance(t2, tmp3))

    print(tmp.i)
    print()
