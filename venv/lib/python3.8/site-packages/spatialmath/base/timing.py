#!/usr/bin/env python3
# -*- coding", t)
"""
Created on Fri Apr 10 14:22:36 2020

@author", t)
"""


import timeit
from ansitable import ANSITable, Column
import numpy as np
from spatialmath import base

N = 100000

table = ANSITable(
    Column("Operation", headalign="^"),
    Column("Time (μs)", headalign="^", fmt="{:.2f}"),
    border="thick",
)


def result(op, t):
    global table

    table.row(op, t / N * 1e6)


# ------------------------------------------------------------------------- #
setup = """
from spatialmath import base
from spatialmath.base import r2q
import numpy as np
s = np.r_[1.0,2,3,4,5,6]
s3 = np.r_[1.0,2,3]
a = np.r_[1.0, 2.0, 3.0]
b = np.r_[-5.0, 4.0, 3.0]
R = base.eul2r(0.1, 0.2, 0.3)
"""
table.rule()
from spatialmath.base import r2q  # , r2q_old


t = timeit.timeit(stmt="q = r2q(R)", setup=setup, number=N)
result("r2q()", t)

# t = timeit.timeit(stmt='q = r2q_old(R)', setup=setup, number=N)
# result("r2q_old()", t)

t = timeit.timeit(stmt="c = np.cross(a,b)", setup=setup, number=N)
result("np.cross()", t)

t = timeit.timeit(stmt="c = base.cross(a,b)", setup=setup, number=N)
result("cross()", t)

# t = timeit.timeit(stmt="a = np.inner(s,s).sum()", setup=setup, number=N)
# result("inner()", t)

# t = timeit.timeit(stmt="a = np.linalg.norm(s) ** 2", setup=setup, number=N)
# result("np.norm**2", t)

# t = timeit.timeit(stmt="a = base.normsq(s)", setup=setup, number=N)
# result("base.normsq", t)

# t = timeit.timeit(stmt="a = (s ** 2).sum()", setup=setup, number=N)
# result("s**2.sum()", t)

# t = timeit.timeit(stmt="a = np.sum(s ** 2)", setup=setup, number=N)
# result("np.sum(s ** 2)", t)

# t = timeit.timeit(stmt="a = np.linalg.norm(s)", setup=setup, number=N)
# result("np.norm(R6)", t)
# t = timeit.timeit(stmt="a = base.norm(s)", setup=setup, number=N)
# result("base.norm(R6)", t)

# t = timeit.timeit(stmt="a = np.linalg.norm(s3)", setup=setup, number=N)
# result("np.norm(R3)", t)

# t = timeit.timeit(stmt="a = base.norm(s3)", setup=setup, number=N)
# result("base.norm(R3)", t)

# t = timeit.timeit(stmt="a = np.isclose(s, s)", setup=setup, number=N)
# result("isclose(s,s)", t)

# t = timeit.timeit(stmt="a = base.iszerovec(s-s)", setup=setup, number=N)
# result("base.iszerovec(s-s)", t)


t = timeit.timeit(stmt="RR=np.eye(4); RR[:3,:3]=R", setup=setup, number=N)
result("RR=np.eye(4); RR[:3,:3]=R", t)

t = timeit.timeit(stmt="RR=np.pad(R, ((0,1),(0,1)), 'constant')", setup=setup, number=N)
result("RR=np.pad(R, ((0,1),(0,1)), 'constant')", t)


table.print()
