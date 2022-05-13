from scipy.spatial.transform import Rotation as R
import scipy, scipy.ndimage
import numpy as np


def pad_with(vector, pad_width, iaxis, kwargs):
  pad_value = 0
  vector[:pad_width[0]] = pad_value
  vector[-pad_width[1]:] = pad_value


def rotate_3d(vol:np.ndarray, M):
  X, Y, Z = vol.shape
  vol_ = vol
  # vol_ = np.pad(vol, ((X,X),(Y,Y),(Z,Z)) , 'constant')
  rotation_matrix = R.from_matrix(M)
  r_x, r_y, r_z = rotation_matrix.as_euler('xyz', degrees=True)
  # print(r_z, r_y, r_x)
  # rotate along z-axis
  vol_ = scipy.ndimage.interpolation.rotate(vol_, r_z, mode='nearest',
                                              axes=(0, 1), reshape=False)
  # rotate along y-axis
  vol_ = scipy.ndimage.interpolation.rotate(vol_, r_y, mode='nearest',
                                              axes=(0, 2), reshape=False)
  # rotate along x-axis
  vol_ = scipy.ndimage.interpolation.rotate(vol_, r_x,
                                                    mode='nearest', axes=(1,2), reshape=False)
  # vol_ = vol_[X:-X,Y:-Y, Z:-Z]
  return vol_


if __name__ == '__main__':
  vol = np.array([[[1,1,1], [2,2,2], [3,3,3]],
                  [[1,1,1], [2,2,2], [3,3,3]],
                  [[1,1,1], [2,2,2], [3,3,3]]])

  M = R.from_euler('xyz', [90,-0,0], degrees=True).as_matrix()
  vol_ = rotate_3d(vol, M)
  print(vol_)
  M_inverse = np.linalg.inv(M)
  print(rotate_3d(vol_, M_inverse))
