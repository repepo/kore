# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""CLI implementation for `conda-env remove`.

Removes the specified conda environment.
"""
from argparse import (
    ArgumentParser,
    Namespace,
    _SubParsersAction,
)


def configure_parser(sub_parsers: _SubParsersAction, **kwargs) -> ArgumentParser:
    from ..auxlib.ish import dals
    from .helpers import (
        add_output_and_prompt_options,
        add_parser_prefix,
        add_parser_solver,
    )

    summary = "Remove an environment."
    description = dals(
        f"""
        {summary}

        Removes a provided environment.  You must deactivate the existing
        environment before you can remove it.

        """
    )
    epilog = dals(
        """
        Examples::

            conda env remove --name FOO
            conda env remove -n FOO

        """
    )

    p = sub_parsers.add_parser(
        "remove",
        help=summary,
        description=description,
        epilog=epilog,
        **kwargs,
    )

    add_parser_prefix(p)
    add_parser_solver(p)
    add_output_and_prompt_options(p)

    p.set_defaults(func="conda.cli.main_env_remove.execute")

    return p


def execute(args: Namespace, parser: ArgumentParser) -> int:
    from ..base.context import context
    from ..cli.main_remove import execute as remove

    args = vars(args)
    args.update(
        {
            "all": True,
            "channel": None,
            "features": None,
            "override_channels": None,
            "use_local": None,
            "use_cache": None,
            "offline": None,
            "force": True,
            "pinned": None,
            "keep_env": False,
        }
    )
    args = Namespace(**args)

    context.__init__(argparse_args=args)

    remove(args, parser)

    return 0
