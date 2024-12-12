from __future__ import absolute_import, print_function, unicode_literals

import argparse
from collections import OrderedDict
import imp
import json
import logging
import os
from os.path import dirname
from pkg_resources import iter_entry_points
import pkgutil
import sys

from clyent.errors import ShowHelp

from ._version import get_versions
from .colors import print_colors

class color(object):
    """
    Deprecated: please use clyent.colors.Color instead
    """
    def __init__(self, text, color_list=()):
        pass

    def __enter__(self):
        pass

    def __exit__(self, err, type_, tb):
        pass

def json_action(action):
    a_data = dict(action._get_kwargs())

    if a_data.get('help'):
        a_data['help'] = a_data['help'] % a_data

    if isinstance(action , argparse._SubParsersAction):
        a_data.pop('choices', None)
        choices = {}
        for choice in action._get_subactions():
            choices[choice.dest] = choice.help
        a_data['choices'] = choices

    reg = {v:k for k, v in action.container._registries['action'].items()}
    a_data['action'] = reg.get(type(action), type(action).__name__)
    if a_data['action'] == 'store' and not a_data.get('metavar'):
        a_data['metavar'] = action.dest.upper()

    a_data.pop('type', None)
    a_data.pop('default', None)

    return a_data

def json_group(group):
    grp_data = {'description': group.description,
                'title': group.title,
                'actions': [json_action(a) for a in group._group_actions if a.help != argparse.SUPPRESS],
                }

    if group._action_groups:
        grp_data['groups'] = [json_group(g) for g in group._action_groups]

    return grp_data

class json_help(argparse.Action):
    def __init__(self, nargs=0, help=argparse.SUPPRESS, **kwargs):
        argparse.Action.__init__(self, nargs=nargs, help=help, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        self.nargs = 0
        docs = {'prog': parser.prog,
                'usage': parser.format_usage()[7:],
                'description': parser.description,
                'epilog': parser.epilog,
               }

        docs['groups'] = []
        for group in parser._action_groups:
            if group._group_actions:
                docs['groups'].append(json_group(group))

        json.dump(docs, sys.stdout, indent=2)
        raise SystemExit(0)

def add_default_arguments(parser, version=None):

    ogroup = parser.add_argument_group('output')
    ogroup.add_argument('--show-traceback', action='store_const', const='always', default='tty',
                        help='Show the full traceback for chalmers user errors (default: %(default)s)')
    ogroup.add_argument('--hide-traceback', action='store_const', const='never', dest='show_traceback',
                        help='Hide the full traceback for chalmers user errors')
    ogroup.add_argument('-v', '--verbose',
                        action='store_const', help='print debug information ot the console',
                        dest='log_level',
                        default=logging.INFO, const=logging.DEBUG)
    ogroup.add_argument('-q', '--quiet',
                        action='store_const', help='Only show warnings or errors the console',
                        dest='log_level', const=logging.WARNING)
    ogroup.add_argument('--color', action='store_const',
                        default='tty', const='always',
                        help='always display with colors')
    ogroup.add_argument('--no-color', action='store_const', dest='color',
                        const='never',
                        help='never display with colors')

    parser.add_argument('--json-help', action=json_help)

    if version:
        parser.add_argument('-V', '--version', action='version',
                            version="%%(prog)s Command line client (version %s)" % (version,))


MODULE_EXTENSIONS = ('.py', '.pyc', '.pyo')

def get_sub_command_names(module):
    return [name for _, name, _ in pkgutil.iter_modules([dirname(module.__file__)]) if not name.startswith('_')]


def get_sub_commands(module):
    names = get_sub_command_names(module)
    this_module = __import__(module.__package__ or module.__name__, fromlist=names)

    for name in names:
        yield getattr(this_module, name)


def add_subparser_modules(parser, module=None, entry_point_name=None):

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    if module:  # LOAD sub parsers from module
        for command_module in get_sub_commands(module):
            command_module.add_parser(subparsers)

    if entry_point_name:  # LOAD sub parsers from setup.py entry_point
        for entry_point in iter_entry_points(entry_point_name):
            add_parser = entry_point.load()
            add_parser(subparsers)

    for key, sub_parser in subparsers.choices.items():
        sub_parser.set_defaults(sub_command_name=key)
        sub_parser.add_argument('--json-help', action=json_help)

def run_command(args, exit=True):

    cli_logger = logging.getLogger('cli-logger')
    cli_logger.error("Command 'chalmers %s'" % getattr(args, 'sub_command_name', '?'))

    try:
        return args.main(args)
    except ShowHelp:
        args.sub_parser.print_help()
        if exit:
            raise SystemExit(1)
        else:
            return 1

__version__ = get_versions()['version']
del get_versions
