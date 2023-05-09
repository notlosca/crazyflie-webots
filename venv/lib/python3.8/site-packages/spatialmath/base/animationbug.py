import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

fig, ax = plt.subplots()
plt.plot([1, 2, 3], [2, 4, 3])


def update(frame):
    print(frame)
    return []


animation = FuncAnimation(
    fig, update, frames=iter(range(10)), blit=False, interval=100, repeat=False
)

FFwriter = FFMpegWriter(fps=100, extra_args=["-vcodec", "libx264"])
animation.save("test.mp4", writer=FFwriter)

print("starting block")
plt.show(block=True)
print("  done with block")

import matplotlib

print(matplotlib.__version__)
print(matplotlib.get_backend())
