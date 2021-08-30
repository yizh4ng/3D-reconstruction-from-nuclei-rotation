import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np

n = 10
frames = 100
x = np.random.random(n)
y = np.random.random(n)
z = np.random.random(n)

zbb = 0. # z_blur_back
zbf = 1. # z_blur_fore

ax = plt.subplot(111)
fig = plt.gcf()

def animate(i):
    global x, y, z
    x += 0.01*(-0.5 + np.random.random(n))
    y += 0.01*(-0.5 + np.random.random(n))
    z += 0.05*(-0.5 + np.random.random(n))

    cmnum = (z-zbf)/(zbb-zbf)
    colors = cm.gray(cmnum) # use hot, ocean, rainbow etc...

    fig = plt.gcf()
    fig.clear()
    ax = plt.gca()
    ax.scatter(x, y, marker='o', s=200., color=colors)
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)

ani = FuncAnimation(fig, animate, frames=60)
ani.save('test.mp4', fps=20)