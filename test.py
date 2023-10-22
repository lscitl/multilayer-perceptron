from typing import overload

class tmp:

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, a: None, b: None) -> None:
        ...

    @overload
    def __init__(self, a: int, b: int) -> None:
        ...

    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b


if __name__ == "__main__":

    t = tmp()

    print(t.a)