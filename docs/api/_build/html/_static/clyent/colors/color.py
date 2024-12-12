import sys
import os
class Color:
    if os.name == 'nt':
        color_map = {

            'yello': 14,
            'blue': 11,
            'green': 10,
            'red': 12,
            'bold': 0,
            'white':15,
            'default': 15

        }
    else:
        color_map = {
            'white': 97,
            'yello': 93,
            'blue': 94,
            'green': 92,
            'red': 91,
            'bold': 1,

            'default': 0,

            'underline': 4,

         }
    def __init__(self, c, file=None):

        if isinstance(c, Color):
            file = file or c.file
            c = c.color_id

        self.file = file or sys.stdout
        self._c = c
        self.current_color_id = None

    @property
    def color_id(self):
        return self.color_map.get(self._c, self._c)

    def __enter__(self):
        set_color = getattr(self.file, 'set_color', None)

        if set_color:
            self.file.flush()
            self.current_color_id = set_color(self.color_id)

        return self

    def __exit__(self, *args):

        set_color = getattr(self.file, 'set_color', None)
        if set_color:
            self.file.flush()
            set_color(self.current_color_id)
