import pandas as pd
import numpy as np

from guess import Guesser
from frame import Frame



class Rotating_Cell():
  def __init__(self, df: pd.DataFrame):
    self.data_frame = df
    self.frame_list = (list(set(list(map(int, df['frame'].values)))))
    self.particle_list = (list(set(list(map(int, df['particle'].values)))))
    self.x, self.y = self.get_xy_position()

  def data_cleasing(self, x: np.array):
    # for every frame, at least k particle
    x = np.delete(x, np.where(np.sum(~np.isnan(x), axis=1) < 3), axis=0)
    # for very particle, at least k frames
    x = np.delete(x.T, np.where(np.sum(~np.isnan(x.T), axis=1) < 15), axis=0).T
    return x

  def get_xy_position(self):
    x = []
    y = []
    frame_list = self.frame_list
    particle_list = self.particle_list
    # for i in range(int(df['frame'].max()) + 1):
    for i in frame_list:
      print(f'read {i} th frame')
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
    return x, y

  def guess_radius(self):
    # center F x 2, cell.x F x P, cell.y F x P
    center = Guesser.guess_center(self.x, self.y)
    # F x 2 x (1 + P)
    R, R_mean = Guesser.guess_radius(self.x, self.y, center)
    self.x, self.y, flag = Guesser.delete_outlier(self.x, self.y, R, R_mean)
    while flag:
      center = Guesser.guess_center(self.x, self.y)
      R, R_mean = Guesser.guess_radius(self.x, self.y, center)
      self.x, self.y, flag = Guesser.delete_outlier(self.x, self.y, R, R_mean)
    self.center = Guesser.smooth_center(center)
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
    for i in range(len(self.frame_list) - 1):
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
    for i in range(len(self.frames) - 1):
      p1 = self.frames[i].points
      p2 = self.frames[i + 1].points
      # for every point
      for j in range(len(p1)):
        if ~np.isnan(p1[j]).any() and np.isnan(p2[j]).any():
          p2[j] = self.frames[i].locale_r @ (p1[j] - self.frames[i].center_position) + \
                  self.frames[i + 1].center_position
          self.frames[i + 1].set_point(p2[j], j)
        for k in range(len(p1)):
          if ~np.isnan(p1[j]).any() and ~np.isnan(p2[k]).any() and j != k:
            if np.linalg.norm(p1[j] - p2[k]) < 5:
              for l in range(i, len(self.frames)):
                self.frames[l].set_point(self.frames[l].points[k], j)
              for l in range(0, len(self.frames)):
                self.frames[l].set_point(np.array([np.nan, np.nan, np.nan]), k)
                # self.frames[l][k] = np.array([np.nan, np.nan, np.nan])

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
        # for k in range(len(p1)):
        #   if ~np.isnan(p1[j]).any() and ~np.isnan(p2[k]).any() and j != k:
        #     if np.linalg.norm(p1[j] - p2[k]) < 5:
        #       p2[j] = p2[k]
        #       p2[k] = np.array([np.nan, np.nan, np.nan])
      self.frames[i - 1].set_points(p2)


  def run(self):
    self.x = self.data_cleasing(self.x)
    self.y = self.data_cleasing(self.y)
    self.x = Guesser.smooth_center(self.x)
    self.y = Guesser.smooth_center(self.y)
    self.frames = []
    self.guess_radius()
    self.radius = 1.7 * self.mean_radius
    self.guess_depth()
    for i in range(len(self.x)):
      self.frames.append(
        Frame(self.x[i], self.y[i], self.z[i], self.center[i], self.radius))
    self.guess_rotation()
    self.attach_rotation()
    self.guess_missing()
    self.guess_missing_back()

if __name__ == '__main__':
  pass