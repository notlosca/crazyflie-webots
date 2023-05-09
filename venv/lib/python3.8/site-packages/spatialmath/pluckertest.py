import numpy as np
import matplotlib.pyplot as plt

import roboticstoolbox as rtb
from spatialmath.base import *
from spatialmath import SE3, Twist3

puma = rtb.models.DH.Puma560()
S, T0 = puma.twists()
print(S)

a = S.line()
print(len(a))
print(a)

puma.plot(puma.qz, block=False)
a.plot("k--", linewidth=1.5, bounds=np.r_[-1, 1, -1, 1, -1, 1.5])
plt.show(block=True)

# from spatialmath import Plucker
# P = [2, 3, 7]
# Q = [2, 1, 0]
# L = Plucker.PQ(P, Q)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
# ax.set_xlim3d(-10, 10)
# ax.set_ylim3d(-10, 10)
# ax.set_zlim3d(-10, 10)
# L.plot(color='red', linewidth=2)

# plt.show()
