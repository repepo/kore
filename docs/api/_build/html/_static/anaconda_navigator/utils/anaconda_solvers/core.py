# -*- coding: utf-8 -*-

"""Core components for solving issues with anaconda config."""

from __future__ import annotations

__all__ = ['SolverCollection', 'POOL']

import collections.abc
import typing

import binstar_client

from anaconda_navigator.utils import notifications
from anaconda_navigator.utils import solvers as common_solvers
from . import utilities


class Solver(typing.Protocol):  # pylint: disable=too-few-public-methods
    """
    Common interface for issue solvers.

    Each solver should:

    - check if there is an issue
    - perform corresponding actions to fix the issue
    - report what was wrong and what was changed
    """

    def __call__(self, configuration: typing.Any) -> notifications.NotificationSource:
        """
        Solve an issue if it is detected.

        :param configuration: Anaconda client configuration to check for issues and fix.
        :return: Message(s) about what was fixed.
        """


class SolverCollection(common_solvers.SolverCollection['Solver']):
    """Collection of issue solvers."""

    __slots__ = ()

    def solve(
            self,
            *,
            tags: str | collections.abc.Iterable[str] | None = None,
    ) -> notifications.NotificationCollection:
        """
        Detect and solve issues for the `context`.

        :param tags: Limit to issue solvers with specific tags.
        :return: Iterator of details about solved issues.
        """
        configuration: typing.Any = {}
        with utilities.catch_and_notify():
            configuration = binstar_client.utils.get_config()

        result: notifications.NotificationCollection = notifications.NotificationCollection()

        solver_record: common_solvers.SolverRecord['Solver']
        for solver_record in self.only(tags=tags):
            result.extend(
                notifications.NOTIFICATION_QUEUE.push(
                    message=solver_record.solver(configuration),
                    caption='Broken Anaconda configuration',
                    tags='anaconda',
                ),
            )

        with utilities.catch_and_notify():
            binstar_client.utils.set_config(configuration)

        return result


POOL: typing.Final[SolverCollection] = SolverCollection()
