import numpy as np

from rotation_fit import rigid_transform_3D

class Guesser():
  def __init__(self, x, y):
    pass

  # Guess set center as the mean of serveral points
  @classmethod
  def guess_center(cls, x:np.array, y:np.array):
    center = []
    # print(x.shape)
    for i in range(len(x)):
      assert np.isnan(x[i]).all() == False
      assert np.isnan(y[i]).all() == False
      center.append([np.nanmean(x[i]), np.nanmean(y[i])])
    return center

  # smooth an array by a convolution
  @classmethod
  def smooth_center(cls, center:np.ndarray):
    center = np.array(center)
    center = np.transpose(center,[1, 0])
    for i in range(len(center)):
      center[i] = np.convolve(np.concatenate(([center[i][0]], center[i],[center[i][-1]])), [1/3, 1/3, 1/3], 'valid')
    center = np.transpose(center, [1, 0])
    return center

  # guess the radius a the mean of radius
  @classmethod
  def guess_radius(cls, x: np.ndarray, y:np.ndarray, center:np.ndarray):
    # everyframe
    R = []
    for i in range(len(x)):
      # every particle
      R_ = []
      for j in range(len(x[0])):
        if np.isnan(x[i][j]):
          r = np.nan
        else:
          r = np.sqrt((x[i][j] - center[i][0]) ** 2 + (y[i][j] - center[i][1]) ** 2)
        R_.append(r)
      R.append(R_)
    return R, np.nanmean(R)

  @classmethod
  # x, y f x p
  def delete_outlier(cls, x:np.ndarray, y:np.ndarray, R, R_mean, de_para = 1.7):
    del_para = de_para
    particle_to_delete = []
    for i in range(len(R)):
      for j in range(len(R[i])):
        if  R[i][j] > del_para * R_mean:
          particle_to_delete.append(j)
    if len(particle_to_delete) != 0:

      x = np.delete(x, particle_to_delete, 1)
      y = np.delete(y, particle_to_delete, 1)
      return  x, y, True
    return x, y, False

  @classmethod
  # x, y f x p
  def guess_z(cls, x:np.ndarray, y:np.ndarray, center, radius):
    depth = radius ** 2 - (x - center[0]) ** 2 - (y - center[1]) ** 2
    depth[depth < 0 ] = 0
    z = np.sqrt(depth)
    return z

  @classmethod
  def find_both_appear_index(cls,x1:np.ndarray,x2:np.ndarray, ):
    assert len(x1) == len(x2)
    index = []
    for i in range(len(x1)):
      if x1[i] == x1[i] and x2[i] == x2[i]:
        index.append(i)
    return index

  @classmethod
  # x, y: (p, 3) center: (2, )
  def guess_rotation(cls, x, y, center_x, center_y):
    index = cls.find_both_appear_index(np.transpose(x)[0], np.transpose(y)[0])
    r, t = rigid_transform_3D(x[index].T, y[index].T, center_x, center_y)
    return r

  @classmethod
  def fit_ellipse(cls, P:np.array):
    from ellipse_fit import EllipsoidTool
    ET = EllipsoidTool()
    (center, radii, rotation) = ET.getMinVolEllipse(P, .01)
    return center, radii, rotation

if __name__ == '__main__':
  pass