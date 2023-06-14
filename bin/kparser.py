#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re

f = open('operators.py','r')
lines = f.readlines()

ops = []
ops2 = []

p_rD    = re.compile("r[1-9]D[1-9]")
p_rhD   = re.compile("r[1-9]h[1-9]D[1-9]")
p_rrhoD = re.compile("r[1-9]rho[1-9]D[1-9]")
p_retaD = re.compile("r[1-9]eta[1-9]D[1-9]")

p_rI    = re.compile("r[1-9]I")
p_rhI   = re.compile("r[1-9]h[1-9]I")
p_rrhoI = re.compile("r[1-9]rho[1-9]I")
p_retaI = re.compile("r[1-9]eta[1-9]I")

p_I     = re.compile("rI[uvfghi]")
p_rD2   = re.compile("rD[1-9]")
p_rhD2  = re.compile("rhD[1-9]")
p_rhI2  = re.compile("rhI")
p_rhI3  = re.compile("r[1-9]hI")
p_rhI4  = re.compile("rh[1-9]I")
p_rhI5  = re.compile("rh[1-9]D")
p_rrhoI2 = re.compile("rrho[1-9]I")
p_rrhoD2 = re.compile("rrho[1-9]D[1-9]")
p_retaD2 = re.compile("retaD[1-9]")
p_retaI2 = re.compile("retaI")

regExpList = [p_rD,p_rhD,p_rrhoD,p_retaD,
              p_rI,p_rhI,p_rrhoI,p_retaI]

regExpList2 = [p_I,p_rD2,p_rhD2,p_rhI2,p_rhI3,p_rhI4,p_rhI5,p_rrhoI2,p_rrhoD2,p_retaD2,p_retaI2]

for iline, line in enumerate(lines):
    for jreg,regExp in enumerate(regExpList):
        m = regExp.findall(line.strip())
        if len(m) > 0:
            ops.append(m)
    for jreg,regExp in enumerate(regExpList2):
        m = regExp.findall(line.strip())
        if len(m) > 0:
            if jreg == 0:
                for k in range(len(m)):
                    m[k] = m[k][:-1]
            ops2.append(m)

ops = [item for subitem in ops for item in subitem]
ops = np.array(ops)
ops = ops.flatten()
ops = np.unique(ops)

ops2 = [item for subitem in ops2 for item in subitem]
ops2 = np.array(ops2)
ops2 = ops2.flatten()
ops2 = np.unique(ops2)

newops = []
lenops = []

for k, op in enumerate(ops):
    lenops.append(len(op))

lenops = np.array(lenops)

ops = ops[np.argsort(-lenops)]
lenops = np.sort(lenops)[::-1]

for k,op in enumerate(ops):

    rx = op[1]
    dx = op[-1]

    if dx == 'I':
        dx = '0'

    if len(op) in [7,8]:
        prof_id = op[2:5]
        prof_dx = op[5]
        subnewop = 'r' + rx + '_' + prof_id + prof_dx + '_D' + dx + '_'
        newops.append(subnewop)
    elif len(op) in [5,6]:
        hx = op[3]
        subnewop = 'r' + rx + '_h' + hx + '_D' + dx + '_'
        newops.append(subnewop)
    elif len(op) in [3,4]:
        subnewop = 'r' + rx + '_D' + dx + '_'
        newops.append(subnewop)

print("Old name, New name")
print("------------------")

for k, op in enumerate(ops):
    print(op,newops[k])

# For ops2
# --------------

newops2 = []
lenops = []

for k, op in enumerate(ops2):
    lenops.append(len(op))

lenops = np.array(lenops)

ops2 = ops2[np.argsort(-lenops)]
lenops = np.sort(lenops)[::-1]


for k,op in enumerate(ops2):

    if op[1].isdigit():
        rx = op[1]
        if op[2] == 'h':
            if op[3].isdigit():
                hx = op[3]
            else:
                hx = '0'
    else:
        rx = '1'
        if op[1] == 'h':
            if op[2].isdigit():
                hx = op[2]
            else:
                hx = '0'


    dx = op[-1]
    if dx == 'I':
        dx = '0'

    if len(op) == 6:
        prof_id = op[1:4]
        if op[4] == 'D':
            prof_dx = '0'
        else:
            prof_dx = op[4]
        subnewop = 'r' + rx + '_' + prof_id + prof_dx + '_D' + dx + '_'
        newops2.append(subnewop)
    elif(len(op)) in [3,4]:
        subnewop = 'r' + rx + '_h' + hx + '_D' + dx + '_'
        newops2.append(subnewop)
    elif len(op) == 2:
        subnewop = 'r' + rx + '_D' + dx + '_'
        newops2.append(subnewop)

newops2 = np.array(newops2)

for k, op in enumerate(ops2):
    print(op,newops2[k])


lines_new = []

for iline, line in enumerate(lines):
    for iop, op in enumerate(ops):
        line = line.replace(op,newops[iop])
    for iop, op in enumerate(ops2):
        line = line.replace(op,newops[iop])
    lines_new.append(line)

# Another pass to take care of Iu, Iv etc.

ops = []
p_rh = re.compile("r[1-9]hD[1-9]")
p_rI = re.compile("r[1-9]hI")
p_I = re.compile("I[uvfghi]")
p_hI = re.compile("hI[uvfghi]")
p_hI2 = re.compile("h[1-9]I")
p_hD = re.compile("hD[1-9]")

regExpList = [p_rh,p_I,p_hI,p_hI2,p_hD]

for iline, line in enumerate(lines_new):
    for jreg,regExp in enumerate(regExpList):
        m = regExp.findall(line.strip())
        if len(m) > 0:
            ops.append(m)

ops = [item for subitem in ops for item in subitem]
ops = np.array(ops)
ops = ops.flatten()
ops = np.unique(ops)

newops = []
lenops = []

for k, op in enumerate(ops):
    lenops.append(len(op))

lenops = np.array(lenops)

ops = ops[np.argsort(-lenops)]
lenops = np.sort(lenops)[::-1]

for k,op in enumerate(ops):

    if len(op) == 5:
        rx = op[1]
        hx = '0'
        dx = op[-1]
        if dx == 'I':
            dx = '0'

        subnewop = 'r' + rx + '_h' + hx + '_D' + dx + '_'
        newops.append(subnewop)
    else:
        rx = '0'
        dx = '0'
        hx = '0'
        if len(op) == 3:
            if op[1] == 'D':
                dx = op[2]
                subnewop = 'r' + rx + '_h' + hx + '_D' + dx + '_'
            elif op[0] == 'h':
                if op[1].isdigit():
                    hx = op[1]
                    subnewop = 'r' + rx + '_h' + hx + '_D' + dx + '_'
                else:
                    subnewop = 'r' + rx + '_h' + hx + '_D' + dx + '_' + op[-1]
            newops.append(subnewop)
        elif len(op) == 2:
            subnewop = 'r' + rx + '_D' + dx + '_' + op[-1]
            newops.append(subnewop)

for k, op in enumerate(ops):
    print(op,newops[k])

lines_new_final = []

for iline, line in enumerate(lines_new):
    for iop, op in enumerate(ops):
        line = line.replace(op,newops[iop])
    lines_new_final.append(line)

g = open('operators_new.py','w')
g.writelines(lines_new_final)
g.close()
f.close()