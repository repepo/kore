# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""This module contains several utilities for Anaconda Navigator package."""

import os
import string
import typing
from urllib import parse
from anaconda_navigator.static.images import IMAGE_PATH
from anaconda_navigator.utils import encoding
from anaconda_navigator.utils.py3compat import is_unicode, u


def get_image_path(filename):
    """Return image full path based on filename."""
    img_path = os.path.join(IMAGE_PATH, filename)

    if os.path.isfile(img_path):
        return img_path
    return None


def try_int(x: typing.Any) -> typing.Any:
    """
    Try converting `x` into :class:`~int`.

    If it is not possible - just return the original `x` value.
    """
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def sort_versions(versions=(), reverse=False, sep='.'):  # pylint: disable=too-many-locals
    """Sort a list of version number strings.

    This function ensures that the package sorting based on number name is
    performed correctly when including alpha, dev rc1 etc...
    """
    if not versions:
        return []

    versions = list(versions)
    new_versions, alpha, sizes = [], set(), set()

    for item in versions:
        it = item.split(sep)
        temp = []
        for i in it:
            x = try_int(i)
            if not isinstance(x, int):
                x = u(x)
                middle = x.lstrip(string.digits).rstrip(string.digits)
                tail = try_int(x.lstrip(string.digits).replace(middle, ''))
                head = try_int(x.rstrip(string.digits).replace(middle, ''))
                middle = try_int(middle)
                res = [item for item in [head, middle, tail] if item != '']
                for r in res:
                    if is_unicode(r):
                        alpha.add(r)
            else:
                res = [x]
            temp += res
        sizes.add(len(temp))
        new_versions.append(temp)

    # replace letters found by a negative number
    replace_dic = {}
    alpha = sorted(alpha, reverse=True)
    if len(alpha):
        replace_dic = dict(zip(alpha, list(range(-1, -(len(alpha) + 1), -1))))

    # Complete with zeros based on longest item and replace alphas with number
    nmax = max(sizes)
    for i, new_version in enumerate(new_versions):
        item = []
        for z in new_version:
            if z in replace_dic:
                item.append(replace_dic[z])
            else:
                item.append(z)

        nzeros = nmax - len(item)
        item += [0] * nzeros
        item += [versions[i]]
        new_versions[i] = item

    new_versions = sorted(new_versions, reverse=reverse)
    return [n[-1] for n in new_versions]


def get_domain_from_api_url(url):  # pylint: disable=missing-function-docstring
    url = parse.urlsplit(url) if isinstance(url, (str, )) else url
    parts = list(url)
    parts[1] = url.netloc.split('api.')[-1]
    parts[2] = url.path.split('/api')[0]
    return parse.urlunsplit(parts)
