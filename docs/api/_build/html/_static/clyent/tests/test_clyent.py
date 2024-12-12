from __future__ import unicode_literals
from argparse import ArgumentParser
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import os
import sys
import unittest

import mock

from clyent import add_subparser_modules
from clyent.colors.color_formatter import print_colors
from clyent.colors import Color, ColorStream

def add_hello_parser(subparsers):
    subparser = subparsers.add_parser('hello')
    subparser.add_argument('world')
    subparser.set_defaults(main=mock.Mock())

class Test(unittest.TestCase):

    def test_add_subparser_modules(self):
        parser = ArgumentParser()

        with mock.patch('clyent.iter_entry_points') as iter_entry_points:

            ep = mock.Mock()
            ep.load.return_value = add_hello_parser
            iter_entry_points.return_value = [ep]
            add_subparser_modules(parser, None, 'entry_point_name')

        args = parser.parse_args(['hello', 'world'])
        self.assertEqual(args.world, 'world')

    @unittest.skipIf(os.name == 'nt', 'Cannot colorize StringIO on Windows')
    def test_color_format(self):
        output = StringIO()
        output.fileno = lambda: -1
        stream = ColorStream(output)

        print_colors('Are you', '{=okay!c:green}', 'Annie?', file=stream)

        value = output.getvalue()
        output.close()

        self.assertEqual('Are you \033[92mokay\033[0m Annie?\n', value)

    @unittest.skipIf(os.name == 'nt', 'Cannot colorize StringIO on Windows')
    def test_color_context(self):
        output = StringIO()
        output.fileno = lambda: -1
        stream = ColorStream(output)

        with Color('red', stream):
            print_colors('ERROR!', file=stream)

        value = output.getvalue()
        output.close()

        self.assertEqual('\033[91mERROR!\n\033[0m', value)


if __name__ == '__main__':
    unittest.main()
