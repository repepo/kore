# -*- coding: utf-8 -*-

"""Helper utilities and shortcuts for the :mod:`~anaconda_navigator.utils.workers`."""

from __future__ import annotations

__all__ = ['parallel_reduce', 'parallel_collect']

import typing

from . import workers


T = typing.TypeVar('T')


EMPTY: typing.Any = object()


def parallel_reduce(
        function: typing.Callable[[T, typing.Any], T],
        tasks: typing.Iterable[workers.TaskWorker],
        initial: T = EMPTY,
) -> T:
    """Run multiple tasks in parallel and reduce their results into a single value."""
    threads: typing.List[workers.TaskThread] = [task.thread() for task in tasks]
    threads.reverse()

    while threads:
        thread: workers.TaskThread = threads.pop()
        thread.wait()
        try:
            current: T = typing.cast(workers.TaskResult, thread.result).result
            if initial is EMPTY:
                initial = current
            else:
                initial = function(initial, current)
        except Exception:
            for thread in threads:
                thread.cancel()
            raise
    if initial is EMPTY:
        raise TypeError('at least one task or initial value must be provided')
    return initial


def list_append(collection: typing.List[T], value: T) -> typing.List[T]:
    """
    Append new value to the list.

    Returns the list to which value was appended.
    """
    collection.append(value)
    return collection


def parallel_collect(tasks: typing.Iterable[workers.TaskWorker]) -> typing.List[typing.Any]:
    """
    Run multiple tasks in parallel and return all their results.

    Results are returned in the same order tasks were provided.
    """
    return parallel_reduce(list_append, tasks, [])
