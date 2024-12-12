from __future__ import print_function
import sys
from contextlib import contextmanager

from clyent.colors.color import Color
from clyent.colors.color_formatter import print_colors
import os

try:
    import win32console
except ImportError:
    std_hdls = {}
else:
    # std_output_hdl will be none if this is a service process
    std_hdls = {
        1: win32console.GetStdHandle(win32console.STD_OUTPUT_HANDLE),
        2: win32console.GetStdHandle(win32console.STD_ERROR_HANDLE)
    }

def initialize_colors(when='tty'):

    assert when in ['tty', 'always', 'never'], when

    if when == 'never':
        return

    if sys.stdout.isatty() or when == 'always':
        sys.stdout = ColorStream(sys.stdout)

    if sys.stderr.isatty() or when == 'always':
        sys.stderr = ColorStream(sys.stderr)


class ColorStream(object):

    def __init__(self, stream):
        self.stream = stream
        self.current_color_id = None
        self.win32_hdl = std_hdls.get(self.stream.fileno())

    def write(self, data):
        n = self.stream.write(data)
        return n

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()

    def set_color(self, color_id):
        last_color_id = self.current_color_id
        if os.name == 'nt':
            if self.win32_hdl:

                if color_id is None:
                    color_id = 15
                self.win32_hdl.SetConsoleTextAttribute(color_id)
        else:
            self.stream.write('\033[%sm' % (color_id or 0))

        self.current_color_id = color_id
        return last_color_id

    @property
    def errors(self):
        return self.stream.errors

    @errors.setter
    def errors(self, val):
        self.stream.errors = val

    @property
    def encoding(self):
        return self.stream.encoding

    @encoding.setter
    def encoding(self, val):
        self.stream.encoding = val

def test():
    initialize_colors()


    with Color('red'):
        print("This is red")
        print_colors('hello {=blue!c:blue}')
        print("This is red again")


if __name__ == '__main__':
    test()
