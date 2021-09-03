import json
import pickle

from lambo.gui.vinci.vinci import DaVinci
import pandas as pd
import numpy as np
class visualize_xyz(DaVinci):
  def __init__(self, size: int = 5, **kwargs):
    # Call parent's constructor
    super(visualize_xyz, self).__init__('Nuclei Visualizer', height=size, width=size)

  def draw_xyz(self, xlist=None, ylist=None, zlist=None):
    fig = self.figure
    ax = self.axes3d
    self.axes3d.view_init(elev=30, azim=0)
    #ax.set_axis_off()
    #fig.set_facecolor('black')
    #ax.set_facecolor('black')
    #ax.grid(False)
    #ax.w_xaxis.pane.fill = False
    #ax.w_yaxis.pane.fill = False
    #ax.w_zaxis.pane.fill = False
    z = np.array(zlist)
    xlist, ylist, zlist = np.array(xlist),np.array(ylist), np.array(zlist)
    ax.plot3D(xlist[z>0],ylist[z>0],zlist[z>0], 'ro', markersize=2)
    ax.plot3D(xlist[z < 0], ylist[z < 0], zlist[z < 0], 'bo', markersize=2)

  def vis_animation(self,x,y,z):
    for i in range(len(x)):
      def visual(x_=x[i], y_=y[i], z_=z[i]):
        self.draw_xyz(xlist=x_, ylist=y_, zlist=z_)
      self.add_plotter(visual)

  def read(self,path):
    f = open(path, 'rb')
    df: pd.DataFrame = pickle.load(f)
    x, y, z = [], [], []
    for f in range(int(df['frame'].max()) + 1):
      x.append(df[df['frame'] == f]['x'].tolist())
      y.append(df[df['frame'] == f]['y'].tolist())
      z.append(df[df['frame'] == f]['z'].tolist())
    return x,y,z

  def read_df(self, df):
    x, y, z = [], [], []
    for f in range(int(df['frame'].max()) + 1):
      x.append(df[df['frame'] == f]['x'].tolist())
      y.append(df[df['frame'] == f]['y'].tolist())
      z.append(df[df['frame'] == f]['z'].tolist())
    return x, y, z

  def read_json(self, js):
    data = np.transpose(np.array(json.load(js)) ,axes=[2,0,1])
    x = data[0]
    y = data[1]
    z = data[2]
    return x, y, z


if __name__ == '__main__':
  vis = visualize_xyz()
  x_g, y_g, z_g = vis.read('data.pkl')
  max_particle = 100
  max_frame = 100
  x_g = np.array(x_g)[0:max_frame, 0:max_particle].tolist()
  y_g = np.array(y_g)[0:max_frame, 0:max_particle].tolist()
  z_g = (np.array(z_g)[0:max_frame, 0:max_particle]).tolist()
  print(x_g,y_g,z_g)
  vis.vis_animation(x_g, y_g, z_g)
  '''def visualzie():
    vis.axes3d.view_init(elev=90, azim=0)
    vis.axes3d.plot3D(x_g[0], y_g[0], z_g[0], 'ro', markersize=2)
  vis.add_plotter(visualzie)'''
  vis.show()
  #vis.add_plotter(visualize_ground_true)
