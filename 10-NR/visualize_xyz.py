import json
import pickle

from lambo.gui.vinci.vinci import DaVinci
import pandas as pd
import numpy as np
from read_pickle import read_to_xyz
class visualize_xyz(DaVinci):
  def __init__(self, size: int = 5, **kwargs):
    # Call parent's constructor
    super(visualize_xyz, self).__init__('Nuclei Visualizer', height=size,
                                        width=size)

    self.keep_3D_view_angle = True

  def read_x_y_data(self, df):
    self.objects = []
    for f in range(int(df['frame'].max()) + 1):
      self.objects.append([df[df['frame'] == f]['x'].tolist(),
                           df[df['frame'] == f]['y'].tolist()])

  def read_x_y_z_data(self, df):
    self.objects = []
    for f in range(int(df['frame'].max()) + 1):
      self.objects.append([df[df['frame'] == f]['x'].tolist(),
                           df[df['frame'] == f]['y'].tolist(),
                           df[df['frame'] == f]['z'].tolist()])

  def read_x_y_z_json(self, data):
    self.objects = np.transpose(np.array(data), axes=[0,2,1]).tolist()

  def draw_2d(self, x, ax):
    #ax.set_xlim(30, 90)
    #ax.set_ylim(130, 190)
    x = np.array(x)[:2]
    ax.scatter(*x, s=5)

  def draw_3d(self, x, ax3d, divide_pos_neg = False):
    # ax3d.set_xlim(-1, 1)
    # ax3d.set_ylim(-1, 1)
    # ax3d.set_zlim(-1, 1)
    if divide_pos_neg:
      x_, y_, z_ = x
      z = np.array(z_)
      xlist, ylist, zlist = np.array(x_), np.array(y_), np.array(z_)
      ax3d.plot3D(xlist[z > 0], ylist[z > 0], zlist[z > 0], 'ro', markersize=2)
      ax3d.plot3D(xlist[z < 0], ylist[z < 0], zlist[z < 0], 'bo', markersize=2)
    else:
      ax3d.scatter(*x, 'bo', s=5)




if __name__ == '__main__':
  '''f = open('data_real.pkl', 'rb')
  df: pd.DataFrame = pickle.load(f)
  df = df[df['frame'] < 10]
  from data_cleasing import remove_unlink
  df = remove_unlink(df)
  # df = df[df['z'] > 0]
  vis = visualize_xyz()
  vis.read_x_y_data(df)
  vis.add_plotter(vis.draw_2d)
  vis.show()'''
  x = json.load(open('./ground_truth/x.json', 'rb'))
  y = json.load(open('./ground_truth/y.json', 'rb'))
  z = json.load(open('./ground_truth/z.json', 'rb'))
  x, y, z = read_to_xyz('./data_Dai.pkl')
  vis = visualize_xyz()
  vis.objects = np.expand_dims(np.array([x[0], y[0], z[0]]),axis=0)
  print(vis.objects.shape)
  vis.add_plotter(vis.draw_3d)
  vis.show()
