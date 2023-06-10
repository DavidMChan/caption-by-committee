from cbc.utils.python import singleton


def test_singleton() -> None:
    @singleton
    class Foo:
        def __init__(self, x: int) -> None:
            self.x = x

    foo1 = Foo(1)
    foo2 = Foo(2)
    assert foo1.x == foo2.x == 1
    assert foo1 is foo2
