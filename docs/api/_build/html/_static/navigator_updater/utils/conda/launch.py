# -*- coding: utf-8 -*-

"""Wrappers to launch and retrieve output of external processes."""

from __future__ import annotations

__all__ = ['ProcessOutput', 'run_process']

import subprocess  # nosec
import typing
from navigator_updater.utils import ansi_utlils, subprocess_utils


class ProcessOutput(typing.NamedTuple):
    """
    Output of external process.

    :param stdout: Content of the general output of the process.
    :param stderr: Content of the error output of the process.
    :param error: Flag that there was an issue running the process.
    """

    stdout: str = ''
    stderr: str = ''
    error: bool = False


def run_process(cmd_list: typing.Sequence[str]) -> ProcessOutput:
    """
    Run subprocess with cmd_list and return stdout, stderr, error.

    :param cmd_list: Command line arguments for the process.
    :return: Collected output of the process.
    """
    stdout: str = ''
    stderr: str = ''
    error: bool = False
    try:
        raw_stdout: bytes
        raw_stderr: bytes
        process: subprocess.Popen
        with subprocess.Popen(  # nosec
                args=cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess_utils.CREATE_NO_WINDOW
        ) as process:
            raw_stdout, raw_stderr = process.communicate()

        stdout = ansi_utlils.escape_ansi(raw_stdout.decode())
        stderr = ansi_utlils.escape_ansi(raw_stderr.decode())
    except OSError:
        error = True

    return ProcessOutput(stdout=stdout, stderr=stderr, error=error)
