#!/usr/bin/env python3
import re
import numpy as np
import sys

'''
Returns a list of all the operators with names matching the convention used in Kore in a file given as argument.
The second list returned gives the parity guessed for each operator. 
Throws a warning message when more than one parity present.
'''

regex = r"(?:(?:r|q)[0-9]_)?(?:h[0-9]_)?(?:(?:\w{3}[0-9]_)*)?(?:D[0-9]_)(?:[uvfghi])"
r = re.compile(regex)

filename = sys.argv[1]
f = open(filename,'r')
lines = f.readlines()

l = []
p = []

for iline, line in enumerate(lines):

    matches = r.finditer(line.strip())
    sorted_matches = sorted(matches, key=lambda m: len(m.group(0)), reverse=True)

    for m in sorted_matches:
        l.append(m.group(0)[:-2]) # append to list and drop section name from and of str

l = np.unique(np.array(l)).tolist()
print(l)

for str in l:
    if sum([int(i) for i in list(map(lambda x: x.replace('dr', '-1'), re.findall('(?:dr|[0-9])',str)))])%2:
        p.append("odd")
    else:
        p.append("even")

print(p)        

if len(np.unique(np.array(p)).tolist())>1:
    print("> Caution: check operators parity")