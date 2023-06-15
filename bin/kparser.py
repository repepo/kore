#!/usr/bin/env python3

import numpy as np
import re

# regex = r"(?P<rpow>r[1-9]?)?(?P<hpow>h[1-9]?)?(?P<prof>\w{3}[1-9]?)?(?P<deriv>D[1-9]|I)(?P<section>[uvfghi])"
regex = r"(?P<rpow>(?:r|q)[1-9]?)?(?P<hpow>h[1-9]?)?(?P<prof>\w{3}[1-9]?)?(?P<deriv>D[1-9]|I)(?P<section>[uvfghi])"
p = re.compile(regex)

f = open('operators.py','r')
lines = f.readlines()
# lines = [lines[210]]
lines_new = []

for iline, line in enumerate(lines):

    matches = p.finditer(line.strip())
    sorted_matches = sorted(matches, key=lambda m: len(m.group(0)), reverse=True)

    for m in sorted_matches:
        old_op = m.group(0)
        mdict = m.groupdict()
        new_op = ''
        if mdict['rpow']:
            if len(mdict['rpow']) == 1:
                new_op += mdict['rpow'] + '1_'
            else:
                new_op += mdict['rpow'] + '_'
        else:
            new_op += 'r0_'

        if mdict['hpow']:
            if len(mdict['hpow']) == 1:
                new_op += 'h0_'
            else:
                new_op += mdict['hpow'] + '_'

        if mdict['prof']:
            if len(mdict['prof']) == 3:
                new_op += mdict['prof'] + '0_'
            else:
                new_op += mdict['prof'] + '_'

        if mdict['deriv']:
            if mdict['deriv'] == 'I':
                new_op += 'D0_'
            else:
                new_op += mdict['deriv'] + '_'

        new_op += mdict['section']

        line = line.replace(old_op,new_op)

    lines_new.append(line)

g = open('operators_new.py','w')
g.writelines(lines_new)
g.close()
f.close()