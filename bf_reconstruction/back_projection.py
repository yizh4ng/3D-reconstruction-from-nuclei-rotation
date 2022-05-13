from bf_reconstruction.util import rotate_3d
import numpy as np
from scipy.spatial.transform import Rotation as R



def back_projection(vol, img, rotation_matrix,debug=True, crop_circle=False):
  rotation_matrix_inv = np.linalg.inv(rotation_matrix)
  # if debug:
  #   angle = np.linalg.norm(R.from_matrix(rotation_matrix).as_rotvec(degrees=True))
  #   print(angle)
  vol_ = vol
  vol_ = rotate_3d(vol_, rotation_matrix_inv)
  X, Y ,Z = vol_shape = vol_.shape
  X_, Y_ = img.shape
  if crop_circle:
    X_cor, Y_cor = np.ogrid[:X_, :Y_]
    img_mask = np.sqrt((X_cor - int(X_/2)) ** 2 + (Y_cor - int(Y_/2)) ** 2) <= int(X_/2)
    img = img_mask * img
  vol_[:, int(0.5 * (X - X_)):int(0.5 * (X - X_)) + X_,
       int(0.5 * (Y - Y_)):int(0.5 * (Y - Y_)) + Y_,] += np.expand_dims(img, 0)
  vol_ = rotate_3d(vol_, rotation_matrix)
  return vol_
