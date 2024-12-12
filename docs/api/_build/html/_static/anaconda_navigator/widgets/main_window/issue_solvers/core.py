# -*- coding: utf-8 -*-

"""Core components for storing and executing issue solvers."""

from __future__ import annotations

__all__ = [
    'SolverCollection',
    'ConfigurationContext', 'CONFIGURATION_POOL',
    'ConflictContext', 'CONFLICT_POOL',
]

import collections.abc
import typing

from anaconda_navigator.utils import notifications
from anaconda_navigator.utils import solvers as common_solvers

if typing.TYPE_CHECKING:
    from anaconda_navigator.api import anaconda_api
    from anaconda_navigator.config import user as user_config


ContextT = typing.TypeVar('ContextT')
ContextT_contra = typing.TypeVar('ContextT_contra', contravariant=True)


class Solver(typing.Protocol[ContextT_contra]):  # pylint: disable=too-few-public-methods
    """
    Common interface for issue solvers.

    Each solver should:

    - check if there is an issue
    - perform corresponding actions to fix the issue
    - report what was wrong and what was changed
    """

    def __call__(self, context: ContextT_contra) -> notifications.NotificationSource:
        """
        Solve an issue if it is detected.

        :param context: Data structure with data that should be checked for issue and updated if one is found.
        :return: Message(s) about what was fixed.
        """


class SolverCollection(common_solvers.SolverCollection['Solver[ContextT]'], typing.Generic[ContextT]):
    """Collection of issue solvers."""

    __slots__ = ()

    def solve(
            self,
            context: ContextT,
            *,
            tags: str | collections.abc.Iterable[str] | None = None,
    ) -> notifications.NotificationCollection:
        """
        Detect and solve issues for the `context`.

        :param context: Context to check for issues and fix them.
        :param tags: Limit to issue solvers with specific tags.
        :return: Iterator of details about solved issues.
        """
        result: notifications.NotificationCollection = notifications.NotificationCollection()

        solver_record: common_solvers.SolverRecord['Solver[ContextT]']
        for solver_record in self.only(tags=tags):
            result.extend(
                notifications.NOTIFICATION_QUEUE.push(
                    message=solver_record.solver(context),
                    caption='Broken Navigator configuration',
                    tags='navigator',
                ),
            )

        return result


class ConfigurationContext(typing.NamedTuple):
    """Context for fixing a general configuration issues."""

    api: 'anaconda_api._AnacondaAPI'
    config: 'user_config.UserConfig'


CONFIGURATION_POOL: typing.Final[SolverCollection[ConfigurationContext]] = SolverCollection()


class ConflictContext(typing.NamedTuple):
    """Context for fixing issues with conflicting Navigator and Conda configurations."""

    api: 'anaconda_api._AnacondaAPI'
    config: 'user_config.UserConfig'
    conda_info: typing.Any


CONFLICT_POOL: typing.Final[SolverCollection[ConflictContext]] = SolverCollection()
