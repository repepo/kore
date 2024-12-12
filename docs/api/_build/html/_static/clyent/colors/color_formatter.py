from __future__ import unicode_literals, print_function
import string
import sys
from .color import Color

class colored_text(object):
    def __init__(self, text):
        self.text = text

class ColorFormatStream(string.Formatter):

    def __init__(self, stream):
        self.stream = stream or sys.stdout

    def convert_field(self, value, conversion):
        if conversion == 'c':
            return colored_text(value)

        rv = string.Formatter.convert_field(self, value, conversion)
        return rv

    def format_field(self, value, format_spec):
        if isinstance(value , colored_text):
            with Color(format_spec, self.stream):
                self.stream.write(value.text)
            return
        else:
            rv = string.Formatter.format_field(self, value, format_spec)
            return rv

    def get_field(self, field_name, args, kwargs):
        if field_name.startswith('='):
            return field_name[1:], None
        else:
            return string.Formatter.get_field(self, field_name, args, kwargs)

    # The signature of _vformat changed in Python 3.4. It added a new argument
    # `auto_arg_index`.
    # https://github.com/python/cpython/blob/3.4/Lib/string.py#L192
    # https://github.com/python/cpython/blob/2.7/Lib/string.py#L567
    if sys.version_info[:2] > (3, 3):
        def _vformat(self, format_string, args, kwargs, used_args, recursion_depth,
                     auto_arg_index=0):
            if recursion_depth < 0:
                raise ValueError('Max string recursion exceeded')
            result = []
            for literal_text, field_name, format_spec, conversion in \
                    self.parse(format_string):

                # output the literal text
                if literal_text:
                    # CHANGED: write instead of append
                    self.stream.write(literal_text)

                # if there's a field, output it
                if field_name is not None:
                    # this is some markup, find the object and do
                    #  the formatting

                    # handle arg indexing when empty field_names are given.
                    if field_name == '':
                        if auto_arg_index is False:
                            raise ValueError('cannot switch from manual field '
                                             'specification to automatic field '
                                             'numbering')
                        field_name = str(auto_arg_index)
                        auto_arg_index += 1
                    elif field_name.isdigit():
                        if auto_arg_index:
                            raise ValueError('cannot switch from manual field '
                                             'specification to automatic field '
                                             'numbering')
                        # disable auto arg incrementing, if it gets
                        # used later on, then an exception will be raised
                        auto_arg_index = False

                    # given the field_name, find the object it references
                    #  and the argument it came from
                    obj, arg_used = self.get_field(field_name, args, kwargs)
                    used_args.add(arg_used)

                    # do any conversion on the resulting object
                    obj = self.convert_field(obj, conversion)

                    # expand the format spec, if needed
                    format_spec, auto_arg_index = string.Formatter._vformat(
                        self, format_spec, args, kwargs,
                        used_args, recursion_depth-1,
                        auto_arg_index=auto_arg_index)

                    # format the object and append to the result
                    # CHANGED: remove append
                    self.format_field(obj, format_spec)

            return ''.join(result), auto_arg_index
    else:
        def _vformat(self, format_string, args, kwargs, used_args, recursion_depth,
                     **extras):
            if recursion_depth < 0:
                raise ValueError('Max string recursion exceeded')
            result = []

            for literal_text, field_name, format_spec, conversion in \
                    self.parse(format_string):

                # output the literal text
                if literal_text:
                    # CHANGED: write instead of append
                    self.stream.write(literal_text)

                # if there's a field, output it
                if field_name is not None:
                    # this is some markup, find the object and do
                    #  the formatting

                    # given the field_name, find the object it references
                    #  and the argument it came from
                    obj, arg_used = self.get_field(field_name, args, kwargs)
                    used_args.add(arg_used)

                    # do any conversion on the resulting object
                    obj = self.convert_field(obj, conversion)

                    # expand the format spec, if needed
                    format_spec = string.Formatter._vformat(self, format_spec, args, kwargs,
                                                            used_args, recursion_depth - 1)

                    # format the object and append to the result
                    # CHANGED: remove append
                    self.format_field(obj, format_spec)

            return ''.join(result)

def print_colors(text='', *args, **kwargs):
    '''
    print_colors(value, ..., sep=' ', end='\n', file=sys.stdout)
    '''

    stream = kwargs.pop('file', sys.stdout)

    end = kwargs.pop('end', '\n')
    sep = kwargs.pop('sep', ' ')
    fmt = ColorFormatStream(stream)

    def write_item(item):
        if isinstance(item, Color):
            with item(stream) as text:
                stream.write(text)
        else:
            fmt.vformat(item, (), kwargs)

    if text:
        write_item(text)

    for text in args:
        stream.write(sep)
        write_item(text)

    stream.write(end)

