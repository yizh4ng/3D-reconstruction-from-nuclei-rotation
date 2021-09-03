import json
import math as m

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
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
    self.anchors = pd.DataFrame({'x':[], 'y':[], 'frame':[], 'particle':[]})
    #initialize the nuleus positions
    self.init_nucleus(5,30)
    #self.add_plotter(self.visualize_nucleus)

  def init_nucleus(self, radius:int, number_anchors:int,**kwargs):
    self.radius = radius
    self.add_anchors(number_anchors)

  def add_anchors(self, num_anchors: int):
    u = np.random.random(num_anchors) * 2 * np.pi
    v = np.random.random(num_anchors) * np.pi
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    for i in range(0, 45):
      pos =  Rx(i * m.pi/360) @ np.array((x,y,z)).tolist()
      for p in range(num_anchors):
        x_, y_, z_ = pos[0][p], pos[1][p], pos[2][p]
        new_row = {'x' : x_, 'y' : y_, 'z' : z_, 'frame' : i, 'particle' : p}
        self.anchors = self.anchors.append(new_row, ignore_index = True)

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
    ax.plot(*(self.anchors[self.anchors['frame'] == angle][['x', 'y', 'z']].
              to_numpy().T), 'wo',markersize=2)

  def visualiza_nucleus_no_back(self,ax3d:Axes3D, angle):
    fig = self.figure
    ax = self.axes3d
    self.axes3d.view_init(elev=90, azim=0)
    ax3d.set_xlim(-1.2, 1.2)
    ax3d.set_ylim(-1.2, 1.2)
    ax3d.set_zlim(-1.2, 1.2)
    ax.set_axis_off()
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.plot(*(self.anchors[
      (self.anchors['frame'] == angle) & (self.anchors['z'] > 0)]
        [['x', 'y', 'z']].to_numpy().T), 'wo', markersize=2)

  # add rotation frame to davinci layers
  def rotate(self):
    for ang in range(0, 45):
      def f(ax3d, angl = ang):
        self.visualiza_nucleus_no_back(ax3d, angle=angl)
        # self.visualize_nucleus(ax3d, angle=angl)
      self.add_plotter(f)

  def save_anchors(self):
   self.anchors.to_pickle('data.pkl')

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

