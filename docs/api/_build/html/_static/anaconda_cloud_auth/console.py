from getpass import getpass
from typing import Any
from typing import Protocol


class TConsole(Protocol):
    @staticmethod
    def print(*args: Any, **kwargs: Any) -> None:
        ...

    @staticmethod
    def input(*args: Any, **kwargs: Any) -> Any:
        ...


class SimpleConsole:
    """
    A very simple console class to mimic the necessary methods we use from rich,
    in case anaconda_cloud_cli is unavailable.
    """

    @staticmethod
    def print(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)

    @staticmethod
    def input(*args: Any, password: bool = False, **kwargs: Any) -> Any:
        if password:
            return getpass(args[0])
        else:
            return input(args[0])


console: TConsole


try:
    from anaconda_cloud_cli import console  # type: ignore

except ImportError:
    console = SimpleConsole()
