import json
import math as m

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lambo.gui.vinci.vinci import DaVinci

from lambo.misc.local import walk
from pandas import DataFrame, Series
from roma import console
from typing import Callable, Optional
def Rz(theta):
  return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def Rx(theta):
  return np.array([[1,              0           , 0],
                   [0,  m.cos(theta), -m.sin(theta)],
                   [0,  m.sin(theta), m.cos(theta)]])


class NRSim(DaVinci):

  def __init__(self, size: int = 5, **kwargs):
    # Call parent's constructor
    super(NRSim, self).__init__('Nuclei Simulator', height=size, width=size)

    self.nucleis = None
    #set the radius of the cell
    self.radius = size
    #initialize the nuleus positions
    self.init_nucleus(5,30)
    #self.add_plotter(self.visualize_nucleus)

  def init_nucleus(self, radius:int, number_anchors:int,**kwargs):
    self.radius = radius
    self.add_anchors(number_anchors)

  def add_anchors(self, num_anchors: int):
    self.anchors = []
    u = np.random.random(num_anchors) * 2 * np.pi
    v = np.random.random(num_anchors) * np.pi
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    for i in range(0, 30):
      self.anchors.append(Rx(i * m.pi/360) @ np.array((x,y,z)))

  def visualize_nucleus(self, ax3d: Axes3D, angle):
    fig = self.figure
    ax = self.axes3d
    ax.set_axis_off()
    #self.axes3d.view_init(elev=30, azim=angle)
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.plot(*self.anchors[angle], 'wo',markersize=2)

  # add rotation frame to davinci layers
  def rotate(self):
    for ang in range(0, 30):
      def f(ax3d, angl = ang):
        self.visualize_nucleus(ax3d, angle=angl)
      self.add_plotter(f)


  def record(self, angle: int):
    print(f'Recording (angle = {angle}) ...')
    anim = animation.FuncAnimation(self.figure, self.rotate,
                                            frames=np.arange(0, 362, 2),
                                            interval=100)
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    print('File exported to ...')

  def save_anchors(self):
    x,y,z = np.array(self.anchors)[:,0,:].tolist(),np.array(self.anchors)[:,1,:].tolist(), np.array(self.anchors)[:,2,:].tolist()
    with open('./z.json', 'w') as f:
      json.dump(z, f)
    with open('./x.json', 'w') as f:
      json.dump(x, f)
    with open('./y.json', 'w') as f:
      json.dump(y, f)

if __name__ == '__main__':
  ns = NRSim()
  #print(np.array(ns.anchors)[:,1,:].shape)
  ns.save_anchors()
  ns.rotate()
  #ns.record()
  '''def vis():
    ns.visualize_nucleus(ns.axes3d,angle=30)
  ns.add_plotter(vis)'''
  ns.show()

