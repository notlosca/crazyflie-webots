import cProfile
import spatialmath as sm


def myfunc():
    X1 = sm.SE3.Rand()
    X2 = sm.SE3.Rand()
    Y = []
    for i in range(1000000):
        Y.append(X1 * X2)


cProfile.run("myfunc")
