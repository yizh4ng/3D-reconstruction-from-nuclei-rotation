import numpy as np

class Frames():
  def __init__(self, frames_: list):
    self.frames = frames_

  # f x p x 3
  @property
  def points(self):
    points = []
    for frame in self.frames:
      points.append(frame.points)
    return np.array(points)

  # f x p x 3
  def set_points(self, points):
    for i, frame in enumerate(self.frames):
      frame.set_points(points[i])

  @classmethod
  def cleasing(cls, frames:list):
    for frame in frames:
      frame.delete_nan()

class Frame():
  def __init__(self, x, y, z, center, radius, rotation=None, ellipse_rotation=None,
               radii = None):
    self.x = x # (p, )
    self.y = y
    self.z = z
    self.center = center #(2,)
    self.radius = radius # float

    self.missing = np.zeros(len(self.x))
    self.rotation = rotation
    self.ellipse_rotation = ellipse_rotation
    self.radii = radii

  # p * 3
  @property
  def points(self):
    x = np.expand_dims(self.x, axis=1)
    y = np.expand_dims(self.y, axis=1)
    z = np.expand_dims(self.z, axis=1)

    return np.concatenate([x, y, z], axis=1)

  @property
  def center_position(self):
    return np.array([self.center[0], self.center[1], 0])

  def attach_rotation(self, r, locale_r):
    self.r = r
    self.locale_r = locale_r

  def set_points(self, points):
    self.x = np.transpose(points)[0]
    self.y = np.transpose(points)[1]
    self.z = np.transpose(points)[2]

  def del_points(self, i):
    self.x = np.delete(self.x, i)
    self.y = np.delete(self.y, i)
    self.z = np.delete(self.z, i)
    self.missing = np.delete(self.missing, i)

  # point 1 x 3
  def set_point(self, point, i):
    self.x[i] = np.transpose(point)[0]
    self.y[i] = np.transpose(point)[1]
    self.z[i] = np.transpose(point)[2]

  def delete_nan(self):
    self.x = self.x[~np.isnan(self.x)]
    self.y = self.y[~np.isnan(self.y)]
    self.z = self.z[~np.isnan(self.z)]