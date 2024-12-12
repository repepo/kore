# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation for `conda doctor` subcommand.
Adds various environment and package checks to detect issues or possible environment
corruption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....base.context import context
from ....cli.helpers import (
    add_parser_help,
    add_parser_prefix,
    add_parser_verbose,
)
from ....deprecations import deprecated
from ... import CondaSubcommand, hookimpl

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


@deprecated(
    "24.3", "24.9", addendum="Use `conda.base.context.context.target_prefix` instead."
)
def get_prefix(args: Namespace) -> str:
    context.__init__(argparse_args=args)
    return context.target_prefix


def configure_parser(parser: ArgumentParser):
    add_parser_verbose(parser)
    add_parser_help(parser)
    add_parser_prefix(parser)


def execute(args: Namespace) -> None:
    """Run registered health_check plugins."""
    print(f"Environment Health Report for: {context.target_prefix}\n")
    context.plugin_manager.invoke_health_checks(context.target_prefix, context.verbose)


@hookimpl
def conda_subcommands():
    yield CondaSubcommand(
        name="doctor",
        summary="Display a health report for your environment.",
        action=execute,
        configure_parser=configure_parser,
    )
