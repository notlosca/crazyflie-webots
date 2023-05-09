#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 07:28:47 2020

@author: corkep
"""

import re
import sys

funcnames = ["sin", "cos", "tan", "atan", "atan2"]


def ind_subs(m):
    var = m.group("var")
    indices = m.group("indices")
    # print(var, indices)
    if var in funcnames:
        return m.group(0)

    ind0 = [str(int(i) - 1) for i in indices.split(",")]

    return var + "[" + ",".join(ind0) + "]"


aref = re.compile(r"(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)\((?P<indices>[0-9,]*)\)")
lcomment = re.compile(r"([ \t]*)%")
rcomment = re.compile(r"%(.*$)")

s = sys.stdin.read()
# s = r'''
#             opt.order = 'zyx';
#             % old ZYX order (as per Paul book)
#             if abs(abs(R(3,1)) - 1) < eps  % when |R31| == 1
#                 % singularity

#                 rpy(1) = 0;     % roll is zero
#                 if R(3,1) < 0
#                     rpy(3) = -atan2(R(1,2), R(1,3));  % R-Y
#                 else
#                     rpy(3) = atan2(-R(1,2), -R(1,3));  % R+Y
#                 end
#                 rpy(2) = -asin(R(3,1));
#             else
#                 rpy(1) = atan2(R(3,2), R(3,3));  % R
#                 rpy(3) = atan2(R(2,1), R(1,1));  % Y

#                 [~,k] = max(abs( [R(1,1) R(2,1) R(3,2) R(3,3)] ));
#                 switch k
#                     case 1
#                         rpy(2) = -atan(R(3,1)*cos(rpy(3))/R(1,1));
#                     case 2
#                         rpy(2) = -atan(R(3,1)*sin(rpy(3))/R(2,1));
#                     case 3
#                         rpy(2) = -atan(R(3,1)*sin(rpy(1))/R(3,2));
#                     case 4
#                         rpy(2) = -atan(R(3,1)*cos(rpy(1))/R(3,3));
#                 end
# '''

sc = re.compile(r";$", flags=re.M)
sc2 = re.compile(r";(.*)$", flags=re.M)

s2 = aref.sub(ind_subs, s)
s2 = lcomment.sub(r"\1#", s2)
s2 = rcomment.sub(r"#\1", s2)
s2 = sc.sub("", s2)
s2 = sc2.sub(r"\1", s2)
s2 = re.sub(r"(atan|atan2|asin|acos|sin|cos|tan)\(", r"math.\1(", s2)

# TODO
# else -> else:
# if XXX -> if XXX:
# [ A B C] [A, B, C

print(s2)
