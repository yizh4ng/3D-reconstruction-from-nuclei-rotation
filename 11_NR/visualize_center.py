import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../lambai")
sys.path.insert(0, '../roma')
from roma import console
from lambo import DaVinci
from frame import Frame, Frames

from tframe import tf
from rotating_cell import Rotating_Cell
from optimize import training
from construct_tensor import frames_to_tensors
sys.path.insert(0, "./src")
class Cell_Visualizer(DaVinci):
  def __init__(self, cell: Rotating_Cell):
    # Call parent's constructor
    super(Cell_Visualizer, self).__init__('vis')
    x = np.expand_dims(cell.x, axis=1)
    y = np.expand_dims(cell.y, axis=1)
    z = np.expand_dims(cell.z, axis=1)
    xy = np.concatenate((x, y, z), axis=1)
    assert xy.shape[1] == 3
    self.x_max, self.x_low = np.nanmax(xy[:,0, :]), np.nanmin(xy[:,0, :])
    self.y_max, self.y_low = np.nanmax(xy[:,1, :]), np.nanmin(xy[:,1, :])
    self.z_max, self.z_low = np.nanmax(xy[:,2, :]), np.nanmin(xy[:,2, :])
    # self.axes.set_xlim(x_low, x_max)
    # self.axes.set_ylim(y_low, y_max)
    self.objects = cell.frames
    self.cell = cell

  def draw_2d_with_center(self, x: np.ndarray, ax):
    ax.scatter(*x, s=5, c='blue')
    ax.scatter(*x[:, 0], s=10, c='red')
    ax.set_xlim(self.x_low, self.x_max)
    ax.set_ylim(self.y_low, self.y_max)

  def draw_frame_2d(self, x, ax):
    assert isinstance(x, Frame)
    for i in range(len(x.x)):
      if x.missing[i] == 1:
        ax.scatter(x.x[i], x.y[i], s=5, c='yellow')
      else:
        ax.scatter(x.x[i], x.y[i], s=5, c='blue')
    # ax.scatter(x.x, x.y, s=5, c='blue')
    ax.scatter([x.center[0]], [x.center[1]], s=10, c='red')
    cir = plt.Circle((x.center[0], x.center[1]), x.radius, color='r', fill=False)
    ax.add_patch(cir)
    width = max (self.x_max - self.x_low, self.y_max - self.y_low)
    ax.set_xlim(self.x_low - 0.1 * width, self.x_low + 1.1 * width)
    ax.set_ylim(self.y_low - 0.1 * width, self.y_low + 1.1 * width)

  def draw_frame_3d(self, x, ax3d):
    assert isinstance(x, Frame)
    assert len(x.missing) == len(x.x)
    for i in range(len(x.x)):
      if x.missing[i] == 1:
        ax3d.scatter(x.x[i], x.y[i], x.z[i], s=5, c='yellow')
      else:
        ax3d.scatter(x.x[i], x.y[i], x.z[i], s=5, c='blue')
    # ax3d.scatter(x.x, x.y, x.z, s=20, c = 'c')
    ax3d.scatter([x.center[0]], [x.center[1]], [0], s=10, c='red')
    width = max(self.x_max - self.x_low, self.y_max - self.y_low)
    ax3d.set_xlim(self.x_low - 0.1 * width, self.x_low + 1.1 * width)
    ax3d.set_ylim(self.y_low - 0.1 * width, self.y_low + 1.1 * width)
    ax3d.set_zlim(self.z_low - 0.6 * width, self.z_low + 0.6 * width)
    plt.axis('off')

    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:50j]
    x_ = 0.98 * x.radius * np.cos(u) * np.sin(v) + x.center[0]
    y_ = 0.98 * x.radius * np.sin(u) * np.sin(v) + x.center[1]
    z_ = 0.98 * x.radius * np.cos(v)
    ax3d.plot_surface(x_, y_, z_, cmap=plt.cm.YlGnBu_r, alpha=0.2, linewidth=0,
                      rstride=1, cstride=1,)

  def draw_rotation(self, x, ax3d):
    assert isinstance(x, Frame)
    x_T = np.transpose(x.r)
    for i in range(3):
      ax3d.plot3D([x_T[i][0],0],
                  [x_T[i][1],0],
                  zs=[x_T[i][2],0])

  def train(self, steps=1):
    self.sess = tf.Session()
    console.show_status('Start training...')
    for i in range(steps):
      console.print_progress(i, total=steps)
      self.update()
      self.refresh()

    console.show_status('Finish training')

  def update(self):
    # Busy computing ...
    frames_ = Frames(self.objects)
    # points = frames_.points
    # x = tf.expand_dims(tf.constant(points[:,:,0]), -1 )
    # y = tf.expand_dims(tf.constant(points[:,:,1]), -1)
    # z = tf.expand_dims(tf.Variable(
    #   initial_value= points[:,:,2], trainable=True), -1)
    points = frames_to_tensors(self.objects)
    self.sess.run(tf.global_variables_initializer())
    # points = tf.concat([x, y, z], axis=-1)
    # points = tf.transpose(points, [1, 2, 0])
    val_loss = training(points, self.sess)
    predcit = self.sess.run(points)
    # print(predcit)
    frames_.set_points(predcit)
    self.objects = frames_.frames
    self.cell.frames = self.objects
    with open(f'./{val_loss}.pkl', 'wb+') as f:
      pickle.dump(self.cell, f)
    # points(len(frames_.frames))

if __name__ == '__main__':
  data = 'adam'
  save = True
  with open(f'./pkl/opt/adam/adam_176.pkl', 'rb') as f:
    cell = pickle.load(f)
  cv = Cell_Visualizer(cell)
  # cv.train()
  cv.add_plotter(cv.draw_frame_2d)
  cv.add_plotter(cv.draw_frame_3d)
  cv.add_plotter(cv.draw_rotation)
  cv.show()
  if save:
    dict = {}
    dict['radius'] = cell.radius
    dict['center'] = cell.center
    x, y, z, center = [], [], [], []
    for f in cell.frames:
      x.append(f.x.tolist())
      y.append(f.y.tolist())
      z.append(f.z.tolist())
      center.append(f.center)
    dict['x'] = x
    dict['y'] = y
    dict['z'] = z
    dict['center'] = center
    with open(f'./results/{data}/cell_176.pkl', 'wb+') as f:
      pickle.dump(dict, f)
    # with open(f'./cell.pkl', 'wb+') as f:
    #   pickle.dump(cell, f)