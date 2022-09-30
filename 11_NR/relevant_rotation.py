import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

# load the pkl from the your disk
# data = 'T5P5_4'
# data = 'three_before_division'
# data = 'adam'
# data = 'data_fast_1'
# data = 'T5P5_4'
# path = f'./cell_class/{data}.pkl'
path = r'Z:\Data to ZHANG Yi\20 Rotated HEK 293T for quantification\Yijin Rotated HEK_20220512P9\Yijin Rotated HEK_20220512P9 Cell 3'
# data = 'Yijin Rotated HEK_20220512P9_Cell1_ICC_RAW_ch00_3d.pkl'
data = 'dummy_3d.pkl'
with open(os.path.join(path, data), 'rb') as f:
  cell = pickle.load(f)

rotation = [0]
local_r = 0
euler_angles = []

# Following codes show the line chart of rotation
# predefine a rotation_axis
#TODO: it may appreciated that we draw angle linechart in three views, e.g. x, y, z :)
rotations = [[0], [0], [0]]
for i, frame in enumerate(cell.frames):
  # if i % 10 != 0: continue
  rotvec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True)
  euler_angles.append(R.from_matrix(frame.locale_r).as_euler('zxy', degrees=True))
  for j, axis in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    rotation_partial_vec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True) @ axis
    local_r = np.sign(rotation_partial_vec) * np.linalg.norm(rotation_partial_vec)
    rotations[j].append(local_r + rotations[j][-1])
x_axis = np.arange(len(rotations[0])) * 339 #TODO:
print(x_axis, '\n',rotations)
plt.plot(x_axis, rotations[0], color='r')
plt.plot(x_axis, rotations[1], color='g')
plt.plot(x_axis, rotations[2], color='b')
# plt.plot(x_axis, rotations[1])
# plt.plot(x_axis, rotations[2])
plt.savefig(rf"{path}\relevant_3.png", dpi=400)
plt.show()


import pandas
df = pandas.DataFrame(x_axis, rotations)
df.to_excel(rf'{path}\relevent_3.xlsx')

rotations = [0]
for i, frame in enumerate(cell.frames):
  # if i % 10 != 0: continue
  rotvec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True)
  euler_angles.append(R.from_matrix(frame.locale_r).as_euler('zxy', degrees=True))
  rotation_vector = R.from_matrix(frame.r).as_rotvec(degrees=True)
  local_r = np.linalg.norm(rotation_vector)
  rotations.append(local_r)
x_axis = np.arange(len(rotations)) * 339 #TODO:
print(x_axis, '\n',rotations)
plt.plot(x_axis, rotations)
# plt.plot(x_axis, rotations[1])
# plt.plot(x_axis, rotations[2])
plt.savefig(rf"{path}\relevant_2.png", dpi=400)
plt.show()


import pandas
df = pandas.DataFrame(x_axis, rotations)
df.to_excel(rf'{path}\relevent_2.xlsx')
