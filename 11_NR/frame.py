import numpy as np



class Frame():
  def __init__(self, x, y, z, center, radius):
    self.x = x # (p, )
    self.y = y
    self.z = z
    self.center = center #(2,)
    self.radius = radius # float

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

  # point 1 x 3
  def set_point(self, point, i):
    self.x[i] = np.transpose(point)[0]
    self.y[i] = np.transpose(point)[1]
    self.z[i] = np.transpose(point)[2]

