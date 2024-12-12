# -*- coding: utf-8 -*-

"""Core interfaces for Conda error solvers."""

from __future__ import annotations

__all__ = ['Solver', 'SimpleSolver', 'SolverCollection', 'POOL']

import abc
import collections.abc
import typing

from anaconda_navigator.utils import notifications
from anaconda_navigator.utils import solvers as common_solvers
from .. import types as conda_types


ErrorT_contra = typing.TypeVar('ErrorT_contra', bound='conda_types.CondaErrorOutput', contravariant=True)


class SolverFunction(typing.Protocol[ErrorT_contra]):  # pylint: disable=too-few-public-methods
    """General form of a function, that could be used as a solver."""

    def __call__(self, error: ErrorT_contra) -> notifications.NotificationSource:
        """
        Solve Conda error.

        :param error: Description of the error to solve.
        :return: Message about what was fixed.
        """


ErrorT = typing.TypeVar('ErrorT', bound='conda_types.CondaErrorOutput')


class Solver(typing.Generic[ErrorT], metaclass=abc.ABCMeta):  # pylint: disable=too-few-public-methods
    """
    Abstract base for all Conda error solvers.

    :param kwargs: Exact values to search for in the error body.
    """

    __slots__ = ('__arguments',)

    def __init__(self, **kwargs: typing.Any) -> None:
        """Initialize new :class:`~ErrorSolver` instance."""
        self.__arguments: typing.Final[collections.abc.Mapping[str, typing.Any]] = dict(kwargs)

    def solve(self, error: 'conda_types.CondaErrorOutput') -> notifications.NotificationSource:
        """
        Attempt solution of the `error`.

        :param error: Description of the error to solve.
        :return: Message about what was fixed. Might be :code:`None` if the solver is not applicable.
        """
        if self._applicable(error=error):
            return self._solve(error=typing.cast(ErrorT, error))
        return None

    def _applicable(self, error: 'conda_types.CondaErrorOutput') -> bool:
        """
        Check if this solver could be used to solve the `error`.

        This method might be overwritten to add custom checks.

        :param error: Description of the error to solve.
        :return: Check result.
        """
        return all(
            error.get(key, None) == value
            for key, value in self.__arguments.items()
        )

    @abc.abstractmethod
    def _solve(self, error: ErrorT) -> notifications.NotificationSource:
        """
        Solve Conda error.

        :param error: Description of the error to solve.
        :return: Message about what was fixed.
        """


class SimpleSolver(Solver[ErrorT], typing.Generic[ErrorT]):  # pylint: disable=too-few-public-methods
    """
    Custom :class:`~ErrorSolver`, which uses external function to solve an error.

    :param __function__: Function to use for error solving.
    :param kwargs: Exact values to search for in the error body.
    """

    __slots__ = ('__function',)

    def __init__(self, __function__: 'SolverFunction[ErrorT]', **kwargs: typing.Any) -> None:
        """Initialize new :class:`~SimpleErrorSolver` instance."""
        super().__init__(**kwargs)
        self.__function: typing.Final[SolverFunction[ErrorT]] = __function__

    def _solve(self, error: ErrorT) -> notifications.NotificationSource:
        """
        Solve Conda error.

        :param error: Description of the error to solve.
        :return: Message about what was fixed.
        """
        return self.__function(error=error)


class SolverCollection(common_solvers.SolverCollection[Solver[typing.Any]]):
    """Collection of solvers."""

    __slots__ = ()

    @typing.overload
    def register_function(
            self,
            function: 'SolverFunction[ErrorT]',
            *,
            filters: collections.abc.Mapping[str, typing.Any] | None = None,
            tags: str | collections.abc.Iterable[str] = (),
            unique_tags: str | collections.abc.Iterable[str] = (),
    ) -> SolverFunction[ErrorT]:
        """Register function as a solver."""

    @typing.overload
    def register_function(
            self,
            function: None = None,
            *,
            filters: collections.abc.Mapping[str, typing.Any] | None = None,
            tags: str | collections.abc.Iterable[str] = (),
            unique_tags: str | collections.abc.Iterable[str] = (),
    ) -> collections.abc.Callable[[SolverFunction[ErrorT]], SolverFunction[ErrorT]]:
        """Register function as a solver."""

    def register_function(self, function=None, filters=None, tags=(), unique_tags=()):
        """
        Register new solver.

        This method can be used as a decorator, as well as direct function. See :meth:`~SolverCollection.register` for
        more details.

        :param function: Function that should be registered.
        :param filters: Values that should be in the error body in order to apply current fix.
        :param tags: Optional collection of common tags.
        :param unique_tags: Optional collection of tags, that should be unique only for this particular solver.
        """
        if filters is None:
            filters = {}

        def wrapper(item: SolverFunction[ErrorT]) -> SolverFunction[ErrorT]:
            self.register(
                solver=SimpleSolver[ErrorT](item, **filters),
                tags=tags,
                unique_tags=unique_tags,
            )
            return item

        if function is None:
            return wrapper
        return wrapper(function)

    def solve(
            self,
            error: conda_types.CondaErrorOutput,
            *,
            tags: str | collections.abc.Iterable[str] | None = None,
    ) -> notifications.NotificationCollection:
        """
        Detect and solve issues for the `context`.

        :param error: Conda error to solve.
        :param tags: Limit to issue solvers with specific tags.
        :return: Iterator of details about solved issues.
        """
        result: notifications.NotificationCollection = notifications.NotificationCollection()

        solver_record: common_solvers.SolverRecord[Solver[typing.Any]]
        for solver_record in self.only(tags=tags):
            result.extend(
                notifications.NOTIFICATION_QUEUE.push(
                    message=solver_record.solver.solve(error),
                    caption='Broken Conda configuration',
                    tags='conda',
                ),
            )

        return result


POOL: typing.Final[SolverCollection] = SolverCollection()
