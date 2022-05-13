from scipy.spatial.transform import Rotation as R
import numpy as np



def equalize_rotation(rotation_list):
  new_rotation_list = []
  current_angle = -999
  index_list = []
  for i, r in enumerate(rotation_list):
    rotation = R.from_matrix(r)
    angle = np.linalg.norm(rotation.as_rotvec(degrees=True))
    if angle - current_angle > 1:
      current_angle = angle
      # print(angle)
      new_rotation_list.append(r)
      index_list.append(i)

  return new_rotation_list, index_list

