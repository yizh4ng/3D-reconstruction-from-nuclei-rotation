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
from scipy.spatial.transform import Rotation as R

color_dict = {0: 'red',
              1: 'green',
              2: 'blue'}


sys.path.insert(0, "./src")
class Cell_Visualizer(DaVinci):
  def __init__(self, cell: Rotating_Cell):
    # Call parent's constructor
    super(Cell_Visualizer, self).__init__('vis')
    p = Frames(cell.frames).points
    # x = np.expand_dims(cell.x, axis=1)
    # y = np.expand_dims(cell.y, axis=1)
    # z = np.expand_dims(cell.z, axis=1)
    xy = np.transpose(p, (0, 2, 1))
    # xy = np.concatenate((x, y, z), axis=1)
    assert xy.shape[1] == 3
    self.x_max, self.x_low = np.nanmax(xy[:,0, :]), np.nanmin(xy[:,0, :])
    self.y_max, self.y_low = np.nanmax(xy[:,1, :]), np.nanmin(xy[:,1, :])
    self.z_max, self.z_low = np.nanmax(xy[:,2, :]), np.nanmin(xy[:,2, :])
    # self.axes.set_xlim(x_low, x_max)
    # self.axes.set_ylim(y_low, y_max)
    self.objects = cell.frames
    self.cell = cell
    self.previous_rotation = None

  def draw_2d_with_center(self, x: np.ndarray, ax):
    ax.scatter(*x, s=5, c='blue')
    ax.scatter(*x[:, 0], s=10, c='red')
    ax.set_xlim(self.x_low, self.x_max)
    ax.set_ylim(self.y_low, self.y_max)

  def draw_frame_2d(self, x, ax):
    assert isinstance(x, Frame)
    for i in range(len(x.x)):
      if x.missing[i] == 1:
        ax.scatter(x.x[i], x.y[i], s=5, c='orange')
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
        ax3d.scatter(x.x[i], x.y[i], x.z[i], s=5, c='orange')
      else:
        ax3d.scatter(x.x[i], x.y[i], x.z[i], s=5, c='blue')
    # ax3d.scatter(x.x, x.y, x.z, s=20, c = 'c')
    ax3d.scatter([x.center[0]], [x.center[1]], [0], s=10, c='red')
    width = max(self.x_max - self.x_low, self.y_max - self.y_low)
    ax3d.set_xlim(self.x_low - 0.1 * width, self.x_low + 1.1 * width)
    ax3d.set_ylim(self.y_low - 0.1 * width, self.y_low + 1.1 * width)
    ax3d.set_zlim(self.z_low - 0.1 * width, self.z_low + 0.925 * width)
    plt.axis('off')

    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
    x_ = 0.98 * x.radius * np.cos(u) * np.sin(v) + x.center[0]
    y_ = 0.98 * x.radius * np.sin(u) * np.sin(v) + x.center[1]
    z_ = 0.98 * x.radius * np.cos(v)
    ax3d.plot_surface(x_, y_, z_, cmap=plt.cm.YlGnBu_r, alpha=0.1, linewidth=0,
                      rstride=1, cstride=1,)

  def draw_frame_3d_ellipse(self,x,ax3d):
    assert isinstance(x, Frame)
    assert len(x.missing) == len(x.x)
    for i in range(len(x.x)):
      if x.missing[i] == 1:
        ax3d.scatter(x.x[i], x.y[i], x.z[i], s=5, c='orange')
      else:
        ax3d.scatter(x.x[i], x.y[i], x.z[i], s=5, c='blue')
    # ax3d.scatter(x.x, x.y, x.z, s=20, c = 'c')
    ax3d.scatter([x.center[0]], [x.center[1]], [0], s=10, c='red')
    width = max(self.x_max - self.x_low, self.y_max - self.y_low)
    ax3d.set_xlim(self.x_low - 0.1 * width, self.x_low + 1.1 * width)
    ax3d.set_ylim(self.y_low - 0.1 * width, self.y_low + 1.1 * width)
    ax3d.set_zlim(self.z_low - 0.1 * width, self.z_low + 1.1 * width)
    plt.axis('off')
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:50j]
    x_ = x.radii[0] * np.cos(u) * np.sin(v)
    y_ = x.radii[1] * np.sin(u) * np.sin(v)
    z_ = x.radii[2] * np.cos(v)
    after_rotation = np.transpose(np.array([x_, y_, z_]), [1,2,0]) \
                      @ x.ellipse_rotation @ x.r.T
    after_rotation = np.transpose(after_rotation, [2, 0, 1])
    x_, y_, z_ = after_rotation[0], after_rotation[1], after_rotation[2]
    x_ = x_ + x.center[0]
    y_ = y_ + x.center[1]
    ax3d.plot_surface(x_, y_, z_, cmap=plt.cm.YlGnBu_r, alpha=0.1, linewidth=0,
                      rstride=1, cstride=1, )

  def draw_rotation(self, x, ax3d):
    assert isinstance(x, Frame)
    r = R.from_matrix(x.locale_r)
    # print(self.object_cursor, r.as_euler('zyx', degrees=True))
    # print(self.object_cursor, r.as_rotvec())
    local_r = r.as_rotvec()
    local_r_norm = local_r * 5
    # ax3d.plot3D([local_r_norm[0],0], [local_r_norm[1], 0], [local_r_norm[2], 0])
    # print(self.object_cursor, np.pi * np.linalg.norm(local_r) * np.sign(local_r @ np.array([0, 0, 1])))
    center = x.center
    # x_T = np.transpose(x.r * x.radius * 2)
    x_T = np.transpose(x.r)
    for i in range(3):
      ax3d.plot3D([x_T[i][0] + center[0],center[0]],
                  [x_T[i][1] + center[1],center[1]],
                  zs=[x_T[i][2],0], c=color_dict[i])
      # ax3d.scatter(x_T[i][0] + center[0], x_T[i][1] + center[1],
      #              x_T[i][2], c=color_dict[i])

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
  # data = 'fast_multi_2'
  data = 'three_before_division'
  with open(f'./cell_class/{data}.pkl', 'rb') as f:
    cell = pickle.load(f)

  rotation = [0]
  local_r = 0

  #Following codes show the line chart of rotation
  for i, frame in enumerate(cell.frames):
    # if i % 10 != 0: continue
    rotvec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True)
    if np.array([0, 0, 1]) @ rotvec > 0:
      local_r += np.linalg.norm(R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
    else:
      local_r += -np.linalg.norm(
        R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
    rotation.append(local_r)
  x_axis = np.arange(len(rotation))
  plt.plot(x_axis, rotation)
  plt.show()
  # Following codes is to enlarge the translation for a better visualization.
  # for i, frame in enumerate(cell.frames):
    # if i % 100 != 0: continue
    # frame.x = frame.x + frame.center[0] * 4
    # frame.y = frame.y + frame.center[1] * 4
    # frame.center = frame.center * 5

  # Following codes show the animation of cell
  cv = Cell_Visualizer(cell)
  cv.add_plotter(cv.draw_frame_2d)
  cv.add_plotter(cv.draw_frame_3d)
  cv.add_plotter(cv.draw_frame_3d_ellipse)
  cv.add_plotter(cv.draw_rotation)
  cv.show()

  # Following codes show point trace
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  # fig = plt.figure(facecolor='white')
  # ax = plt.axes(frameon=False)
  line = []
  rotation = [[],[],[]]
  for i, frame in enumerate(cell.frames):
    # if i % 50 == 0:
      # frame.x = frame.x + frame.center[0] * 4
      # frame.y = frame.y + frame.center[1] * 4
      # frame.z = frame.z * 5
      # frame.center = frame.center * 5
      # cv.draw_frame_3d_ellipse(frame, ax)
      # cv.draw_rotation(frame, ax)
    # cv.draw_rotation(frame, ax)
    center = frame.center
    x_T = np.transpose(frame.r * frame.radius * 2)
    if i % 1 == 0:
      # for i in range(3):
        # rotation[i].append([x_T[i][0] + center[0], x_T[i][1] + center[1], x_T[i][2]])
      rotation[0].append([frame.x[0]-center[0], frame.y[0] - center[1],  frame.z[0]])
      rotation[1].append([frame.x[0]-center[0], frame.y[0] - center[1],  0])

  for i in range(3):
    if len(rotation[i]) == 0: continue
    x, y, z = np.transpose(rotation[i]) * 0.065
    N = len(x)
    for j in range(N - 1):
      if i == 1:
        # ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
        #         color=(j/N, j/N, j/N), alpha=0.7)
        ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet(j / N), alpha=1.0)
      else:
        ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet(j / N), alpha=1)
    # ax.plot3D(*np.transpose(rotation[i]), c=color_dict[i])
  plt.show()
  '''ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  # ax.set_xlabel('x')
  # ax.set_ylabel('y')
  # ax.get_yaxis().set_ticks([])
  # ax.get_xaxis().set_ticks([])
  x_ticks = ax.get_xticks()
  y_ticks = ax.get_yticks()
  z_ticks = ax.get_zticks()



  x = [x_ticks[0], x_ticks[0], x_ticks[-1], x_ticks[-1]]
  y = [y_ticks[0], y_ticks[-1], y_ticks[-1], y_ticks[0]]
  z = [0, 0 ,0 ,0]
  from matplotlib.patches import FancyArrowPatch
  from mpl_toolkits.mplot3d import proj3d

  class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  poly = Poly3DCollection([list(zip(x,y,z))], alpha=0.2,color='lightgray')
  ax.add_collection3d(poly)
  for y1 in y_ticks:
    x1, x2 = x_ticks[0], x_ticks[-1]
    ax.plot([x1, x2], [y1, y1], [0, 0], color='lightgray', alpha=0.5)
  for x1 in x_ticks:
    y1, y2 = y_ticks[0], y_ticks[-1]
    ax.plot([x1, x1], [y1, y2], [0, 0], color='lightgray', alpha=0.5)
  # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
  ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0))
  ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0))
  ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0))
  ax.grid(False)
  a = Arrow3D([x_ticks[0], x_ticks[2]], [y_ticks[0], y_ticks[0]],
              [z_ticks[0], z_ticks[0]], mutation_scale=3,
              lw=2, arrowstyle="->", color="black", shrinkA=0)
  ax.add_artist(a)
  a = Arrow3D([x_ticks[0], x_ticks[0]], [y_ticks[0], y_ticks[2]],
              [z_ticks[0], z_ticks[0]], mutation_scale=3,
              lw=2, arrowstyle="->", color="black", shrinkA=0)
  ax.add_artist(a)
  a = Arrow3D([x_ticks[0], x_ticks[0]], [y_ticks[0], y_ticks[0]],
              [z_ticks[0], z_ticks[2]], mutation_scale=3,
              lw=2, arrowstyle="->", color="black", shrinkA=0)
  ax.add_artist(a)
  # ax.get_yaxis().set_ticks([])
  # ax.get_xaxis().set_ticks([])
  # ax.get_zaxis().set_ticks([])

  # plt.grid(color='white', linestyle='-.', linewidth=0.7)
  # plt.colorbar()
  for i in range(3):
    if len(rotation[i]) == 0: continue
    x, y, z = np.transpose(rotation[i]) * 0.065
    N = len(x)
    for j in range(N - 1):
      if i == 1:
        # ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
        #         color=(j/N, j/N, j/N), alpha=0.7)
        ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet(j / N), alpha=0.3)
      else:
        ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet(j / N))
    # ax.plot3D(*np.transpose(rotation[i]), c=color_dict[i])
  plt.show()'''