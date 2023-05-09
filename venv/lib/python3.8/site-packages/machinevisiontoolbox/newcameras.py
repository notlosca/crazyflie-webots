import numpy as np
import scipy as sp
from math import pi
from spatialmath import SE3
from spatialmath.base import e2h, h2e, plot_sphere
from machinevisiontoolbox import *

u0 = 528.1214; v0 = 384.0784; l = 2.7899; m = 996.4617;

im_fisheye = Image('fisheye_target.png', dtype='float', grey=True)
im_fisheye.disp()


n = 500
theta_range = np.linspace(0, pi, n)
phi_range = np.linspace(-pi, pi, n)

Phi, Theta = np.meshgrid(phi_range, theta_range)

r = (l + m) * np.sin(Theta) / (l - np.cos(Theta))
Us = r * np.cos(Phi) + u0
Vs = r * np.sin(Phi) + v0

im_spherical = im_fisheye.interp2d(Us, Vs)
# im_spherical = f(theta_range, phi_range)

im_spherical.disp(badcolor='red')
# plt.show(block=True)


# sphere
R = 1
x = R * np.sin(Theta) * np.cos(Phi)
y = R * np.sin(Theta) * np.sin(Phi)
z = R * np.cos(Theta)

# create 3d Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = im_spherical.colorize()
ax.plot_surface(x.T, y.T, z.T, facecolors=img.image, cstride=1, rstride=1) # we've already pruned ourselves

for az in np.arange(-180, 180, 30):
    for el in np.arange(-180, 180, 30):
        print(az, el)
        ax.view_init(azim=az, elev=el)
        plt.show()
        plt.savefig(f"az{az}el{el}.png")