import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "../../lambai")
sys.path.insert(0, '../roma')
from roma import console

from guess import Guesser
from frame import Frame, Frames


class Rotating_Cell():
  def __init__(self, df: pd.DataFrame, del_para = 1.7, radius_para = 2,
               iteratively_op_radius= True, iterative_times = 2, iterate=2):
    df = df[df['frame'] == df['frame']]
    self.data_frame = df
    self.frame_list = (list(set(list(map(int, df['frame'].values)))))
    self.particle_list = (list(set(list(map(int, df['particle'].values)))))
    self.x, self.y = self.get_xy_position()
    self.del_para = del_para
    self.iterative = iteratively_op_radius
    self.iterative_times = iterative_times
    self.missing = np.zeros(self.x.shape)
    self.radius_para = radius_para
    self.iterate = iterate

  def data_cleasing(self, x: np.array):
    for i in range(3):
      # for very particle, at least k frames
      x = np.delete(x.T, np.where(np.sum(~np.isnan(x.T), axis=1) < 4), axis=0).T
      # for every frame, at least k particle
      x = np.delete(x, np.where(np.sum(~np.isnan(x), axis=1) < 4), axis=0)
    self.missing = np.zeros(x.shape)
    assert len(x[0]) > 2
    return x

  def get_xy_position(self):
    x = []
    y = []
    frame_list = self.frame_list
    particle_list = self.particle_list
    # for i in range(int(df['frame'].max()) + 1):
    console.show_status('Loading DataFrame...')

    for i in frame_list:
      console.print_progress(i, total=len(frame_list))
      x_ = np.array([])
      y_ = np.array([])
      for j in particle_list:
        row = self.data_frame[(self.data_frame['frame'] == i)
                              & (self.data_frame['particle'] == j)]
        if len(row) == 0:
          # print(f"particle{j} at frame {i} disappears")
          x_ = np.concatenate((x_, [np.NAN]))
          y_ = np.concatenate((y_, [np.NAN]))
        else:
          assert len(row) == 1
          row = row.iloc[0]
          # print(f"particle{j} at frame {i}at{row['x'], row['y']}")
          x_ = np.concatenate((x_, [row['x']]))
          y_ = np.concatenate((y_, [row['y']]))
      x.append(x_)
      y.append(y_)
    x = np.array(x)
    y = np.array(y)

    console.show_status('Finish Reading DataFrame.')
    return x, y

  def guess_radius(self):
    # center F x 2, cell.x F x P, cell.y F x P
    center = Guesser.guess_center(self.x, self.y)
    # F x 2 x (1 + P)
    R, R_mean = Guesser.guess_radius(self.x, self.y, center)
    self.x, self.y, flag = Guesser.delete_outlier(self.x, self.y, R, R_mean)
    iter = 1
    if self.iterative:
        while flag and iter <= self.iterative_times:
          print(f'{iter}th iteration on optimizing radius')

          center = Guesser.guess_center(self.x, self.y)
          R, R_mean = Guesser.guess_radius(self.x, self.y, center)
          self.x, self.y, flag = Guesser.delete_outlier(self.x, self.y, R, R_mean)
          # self.x = self.data_cleasing(self.x)
          # self.y = self.data_cleasing(self.y)
          self.missing = np.zeros_like(self.x)
          assert (len(self.x) != 0)
          iter += 1

    self.center = Guesser.guess_center(self.x, self.y)
    self.center = Guesser.smooth_center(self.center)
    R, R_mean = Guesser.guess_radius(self.x, self.y, center)
    self.radius = R
    self.mean_radius = R_mean

  def guess_depth(self):
    self.z = []
    for i in range(len(self.x)):
      z_ = Guesser.guess_z(self.x[i], self.y[i], self.center[i],self.radius)
      self.z.append(z_)

  def guess_rotation(self):
    self.rotation = []
    for i in range(len(self.frames) - 1):
      self.rotation.append(Guesser.guess_rotation(self.frames[i].points,
                                                  self.frames[i + 1].points,
                                                  self.frames[i].center,
                                                  self.frames[i + 1].center))

  def attach_rotation(self):
    current_r = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    for i in range(len(self.frames) - 1):
      self.frames[i].attach_rotation(current_r, self.rotation[i])
      current_r = self.rotation[i] @ current_r
    self.frames[-1].attach_rotation(current_r, self.rotation[-1])

  def guess_missing(self):
    # for evrey frame
    points_to_delete = []
    for i in range(len(self.frames) - 1):
      p1 = self.frames[i].points
      p2 = self.frames[i + 1].points
      # for every point
      for j in range(len(p1)):
        # handle wrong connection
        if ~np.isnan(p1[j]).any():
          if self.missing[i][j] != 1:
            self.missing[i][j] = 0

        if np.isnan(p1[j]).any():
          # if self.missing[i][j] != 1:
          self.missing[i][j] = 1

        # if ~np.isnan(p1[j]).any() and ~np.isnan(p2[j]).any():
        #   if np.linalg.norm(p1[j] - p2[j]) > 15:
        #     self.frames[i + 1].set_point(np.array([np.nan, np.nan, np.nan]), j)
        #     # p2 = self.frames[i + 1].points
        #     self.missing[i + 1][j] = 1

        # predict position
        if ~np.isnan(p1[j]).any() and np.isnan(p2[j]).any():
          p2[j] = self.frames[i].locale_r @ (p1[j] - self.frames[i].center_position) + \
                  self.frames[i + 1].center_position
          self.frames[i + 1].set_point(p2[j], j)
          self.missing[i + 1][j] = 1

        # connect division
    # for i in range(len(self.frames) - 1):
    #   p1 = self.frames[i].points
    #   p2 = self.frames[i + 1].points
    #   for j in range(len(p1)):
    #     for k in range(0, len(p1)):
    #       if ~np.isnan(p1[j]).any() and ~np.isnan(p2[k]).any() and j != k:
    #         if np.linalg.norm(p1[j][:2] - p2[k][:2]) < 3:
    #           if len(np.argwhere(np.isnan(Frames(self.frames).points[:i, j]))) == 0:
    #             to_append = j
    #             to_delete = k
    #           elif len(np.argwhere(np.isnan(Frames(self.frames).points[:i, k]))) == 0:
    #             to_append = k
    #             to_delete = j
    #           elif np.sum(self.missing[:i + 1, k] == 0) > np.sum(self.missing[:i + 1, j] == 0):
    #             to_append = j
    #             to_delete = k
    #           else:
    #             to_append = k
    #             to_delete = j
    #           for l in range(i + 1, len(self.frames)):
    #             self.frames[l].set_point(self.frames[l].points[to_delete], to_append)
    #             self.missing[l][to_append] = self.missing[l][to_delete]
    #           for l in range(0, len(self.frames)):
    #             self.frames[l].set_point(np.array([np.nan, np.nan, np.nan]), to_delete)
    #           points_to_delete.append(to_delete)

    points_to_delete = list(set(points_to_delete))
    # print(points_to_delete)
    for l in range(0, len(self.frames)):
      self.frames[l].del_points(points_to_delete)
    self.missing = np.delete(self.missing, points_to_delete, axis=1)
    # points = Frames(self.frames).points[:,:,0]
    assert self.missing.shape ==  (len(self.frames), len(self.frames[0].points))

  def guess_missing_back(self):
    for i in range(len(self.frames) - 1, 0, -1):
      p1 = self.frames[i].points
      p2 = self.frames[i - 1].points
      # for every point
      for j in range(len(p1)):
        if ~np.isnan(p1[j]).any() and np.isnan(p2[j]).any():
          p2[j] = self.frames[i - 1].locale_r.T @ (
                p1[j] - self.frames[i].center_position) + \
                  self.frames[i - 1].center_position

          self.missing[i - 1][j] = 1
        # for k in range(len(p1)):
        #   if ~np.isnan(p1[j]).any() and ~np.isnan(p2[k]).any() and j != k:
        #     if np.linalg.norm(p1[j] - p2[k]) < 5:
        #       p2[j] = p2[k]
        #       p2[k] = np.array([np.nan, np.nan, np.nan])
      self.frames[i - 1].set_points(p2)

    # points = Frames(self.frames).points[:,:,0]
    # assert np.isnan(Frames(self.frames).points).any() == False

  def attaching_missing(self):
    assert self.missing.shape ==  (len(self.frames), len(self.frames[0].points))
    for i in range(len(self.missing)):
      for j in range(len(self.missing[0])):
        self.frames[i].missing[j] = self.missing[i][j]

  def smooth(self):
    FRAMES = Frames(self.frames)
    # f x p x 3
    points = FRAMES.points
    # f x p
    x = points[:,:,0]
    y = points[:,:,1]
    z = points[:,:,2]
    x_ = Guesser.smooth_center(x)
    y_ = Guesser.smooth_center(y)
    z_ = Guesser.smooth_center(z)
    new_points = np.concatenate((np.expand_dims(x_, axis=-1),
                                np.expand_dims(y_, axis=-1),
                                np.expand_dims(z_, axis=-1)), axis=-1)

    FRAMES.set_points(new_points)
    self.frames = FRAMES.frames

  def guess_ellipse(self):
    points = np.concatenate((self.frames[0].points - np.array([self.center[0][0],
                                                               self.center[0][1],
                                                               0]),
                             -(self.frames[0].points - np.array([self.center[0][0],
                                                               self.center[0][1],
                                                               0]))),
                            axis=0)
    center, radii, rotation = Guesser.fit_ellipse(points)
    self.radii = radii
    self.ellipse_direciton = rotation
    for f in self.frames:
      f.ellipse_rotation = rotation
      f.radii = radii

  def run(self):
    for i in range(self.iterate):
      # self.x = self.data_cleasing(self.x)
      # self.y = self.data_cleasing(self.y)
      # self.x = Guesser.smooth_center(self.x)
      # self.y = Guesser.smooth_center(self.y)
      # self.x = self.data_cleasing(self.x)
      # self.y = self.data_cleasing(self.y)
      self.frames = []
      self.guess_radius()
      if self.radius_para is None:
        self.radius = self.del_para * self.mean_radius
      else:
        self.radius = self.radius_para * self.mean_radius
      self.guess_depth()
      for i in range(len(self.x)):
        self.frames.append(
          Frame(self.x[i], self.y[i], self.z[i], self.center[i], self.radius))
      assert len(self.frames) > 0
      self.guess_rotation()
      self.attach_rotation()
      self.guess_missing()
      self.guess_missing_back()
      self.attaching_missing()
      self.guess_ellipse()
      self.x = Frames(self.frames).points[:,:,0]
      self.y = Frames(self.frames).points[:,:,1]
      self.z = Frames(self.frames).points[:,:,2]

if __name__ == '__main__':
  np.expand_dims
  pass