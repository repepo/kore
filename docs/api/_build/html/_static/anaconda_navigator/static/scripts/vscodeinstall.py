# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

"""VSCode linux installation helper script."""

import io
import os
import sys


LINUX = sys.platform.startswith('linux')


if __name__ == '__main__':
    args = sys.argv[1:]

    fpath = None
    data = None
    status = 0

    if args and LINUX:
        distro = args[0].lower()

        if distro in ['centos', 'rhel', 'fedora']:
            fpath = '/etc/yum.repos.d/vscode.repo'
            data = """[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc"""
        elif distro in ['opensuse']:
            fpath = '/etc/zypp/repos.d/vscode.repo'
            data = """[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
type=rpm-md
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc"""

    if fpath and data:
        path = os.path.dirname(fpath)
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
            except Exception:  # pylint: disable=broad-except
                status += 1

        try:
            with io.open(fpath, 'w', encoding='utf-8') as fh:
                fh.write(data)
        except Exception:  # pylint: disable=broad-except
            status += 1

    sys.exit(status)
