#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Helper utility to clean up json bundled content."""

import json
import sys


def verify(content):
    """Verify the content keys."""
    for item in content:
        for key in 'tags title summary banner uri image_file_path'.split():
            if key not in item:
                name = item.get('title')
                if name is None:
                    name = item.get('uri')
                if name is None:
                    name = 'UNKNOWN'
                print(name, 'is missing key', key)


if __name__ == '__main__':
    for fname in sys.argv[1:]:
        with open(fname) as f:  # pylint: disable=unspecified-encoding
            fcontent = json.load(f)
        if 'links.json' in fname:
            verify(fcontent)
        with open(fname, 'w') as out:  # pylint: disable=unspecified-encoding
            out.write(json.dumps(fcontent, indent=2))
