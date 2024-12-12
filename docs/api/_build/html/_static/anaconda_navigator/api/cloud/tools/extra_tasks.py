# -*- coding: utf-8 -*-

"""Collection for additional tasks for environment management and interaction with Cloud."""

__all__ = ['clear_environment']

import typing
import yaml
from anaconda_navigator.utils import workers


@workers.Task
def clear_environment(source_file: str, target_file: str, name: str) -> None:
    """
    Prepare a new cleared environment in `target_file` from the one in the `source_file`.

    This method replaces :code:`name: null` with the actual environment name, and removes `prefix: ...`.

    :param source_file: `conda env export` result, that should be changed.
    :param target_file: New file to create from the `source_file` with cleanups.
    :param name: Name of the environment to inject into `source_file`.
    """
    replacements: typing.Final[typing.Mapping[str, str]] = {
        yaml.dump({'name': None}): yaml.dump({'name': name}),
    }

    source: typing.TextIO
    target: typing.TextIO
    with open(source_file, 'rt', encoding='utf-8') as source, open(target_file, 'wt', encoding='utf-8') as target:
        line: str
        for line in source:
            if line.startswith('prefix:'):
                continue
            target.write(replacements.get(line, line))
