#!/usr/bin/env python3
'''
Sets one parameter in bin/parameters.py. Use as

    python3 tools/setpar.py name value [file]

value : any python expression, written as it is: par.name = value
file  : default bin/parameters.py

NEW (2026-11-08): the assignment takes the place of each statement par.name = ... of the file (its comment is kept), or,
if there is none, becomes the file's last line, after par.set_scales() and the other values, so that it holds.
tools/ramp.sh and tools/dodirs.sh use it. A parameter that only par.set_scales() reads (timescale, Ek, Etherm, Ecomp,
Le) changes nothing when no call of it follows: then it warns, and the parameters derived from it, such as ViscosD, are
the ones to set.
'''
import ast
import sys

if len(sys.argv) not in (3, 4):
    sys.exit(__doc__)
name, value = sys.argv[1:3]
fname = sys.argv[3] if len(sys.argv) == 4 else 'bin/parameters.py'
new = ('par.%s = %s' % (name, value)).encode()
src = open(fname, 'rb').read()
try:
    ast.parse(new)
    tree = ast.parse(src)
except SyntaxError as e:
    sys.exit('tools/setpar.py: %s' % e)

# the statements par.name = ... (one target), as byte spans: ast counts columns in bytes
spans = [(n.lineno, n.col_offset, n.end_lineno, n.end_col_offset) for n in tree.body
         if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
         and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == 'par' and n.targets[0].attr == name]
lines = src.splitlines(keepends=True)
if spans:
    for l0, c0, l1, c1 in reversed(spans):
        lines[l0 - 1:l1] = [lines[l0 - 1][:c0] + new + lines[l1 - 1][c1:]]
    src = b''.join(lines)
else:
    src = src.rstrip(b'\n') + b'\n' + new + b'\n'
open(fname, 'wb').write(src)

if name in ('timescale', 'Ek', 'Etherm', 'Ecomp', 'Le'):   # read by set_scales() alone
    body = ast.parse(src).body
    last = max(n.lineno for n in body if isinstance(n, ast.Assign) and len(n.targets) == 1
               and ast.unparse(n.targets[0]) == 'par.' + name)
    if not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'par.set_scales'
               and n.lineno > last for n in body):
        print('tools/setpar.py: warning: only par.set_scales() reads par.%s, and %s does not call it after line %d: the run '
              'will not see it; set the parameters derived from it (ViscosD, ThermaD, CompoD, Gaspard) instead'
              % (name, fname, last), file=sys.stderr)
