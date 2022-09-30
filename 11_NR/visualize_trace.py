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
abs_rotation = True
abs_indicator = 1 if abs_rotation else -1

# Following codes show the line chart of rotation
# predefine a rotation_axis
#TODO: it may appreciated that we draw angle linechart in three views, e.g. x, y, z :)
rotation_axis = np.array([0, 0, 1])
for i, frame in enumerate(cell.frames):
  # if i % 10 != 0: continue
  rotvec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True)
  euler_angles.append(R.from_matrix(frame.locale_r).as_euler('zxy', degrees=True))
  # if abs_rotation:
  #   local_r += np.abs(np.linalg.norm(
  #     R.from_matrix(frame.locale_r).as_rotvec(degrees=True)))
  # else:
  #   local_r = np.linalg.norm(
  #     R.from_matrix(frame.r).as_rotvec(degrees=True))
  if abs_rotation:
    local_r += np.abs(np.linalg.norm(
      R.from_matrix(frame.locale_r).as_rotvec(degrees=True)))
  else:
    if rotation_axis @ rotvec > 0:
      local_r += np.linalg.norm(
        R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
    else:
      local_r -= np.linalg.norm(
        R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
  rotation.append(local_r)
x_axis = np.arange(len(rotation)) * 339 #TODO:
print(x_axis, '\n',rotation)
import pandas
df = pandas.DataFrame(x_axis, rotation)
# save_path = rf'Z:\Data to ZHANG Yi\figure_modification\{data}'
# import os
# if not os.path.exists(save_path):
#   os.mkdir(save_path)
plt.plot(x_axis, rotation)
if abs_rotation:
  df.to_excel(rf'{path}\abs.xlsx')
  plt.draw()
  plt.savefig(rf"{path}\abs.png", dpi=400)
else:
  df.to_excel(rf'{path}\relevent.xlsx')
  plt.savefig(rf'{path}\relevent.png', dpi=400)
pandas.DataFrame(euler_angles).to_excel(rf'{path}\euler_angle.xlsx')
plt.show()

# following codes are to visualize point traces.
# You specify the point index you would like to visualize
# point_index = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14]
point_index = [0]
# Set to Ture if you want to draw the rotation axis trace instead of the point
vis_rotation_axis = False
# Set to Ture if you want to draw trace with translation information
with_translation = False
if vis_rotation_axis:
  point_index = [0]

matplotlib.use('tkagg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line = []
rotation = []
for i, frame in enumerate(cell.frames):
  # if i % 1 != 0: continue
  center = frame.center
  if vis_rotation_axis:
    x_T = np.transpose(frame.r * frame.radius * 2)
  else:
    # x_T = np.transpose([frame.x, frame.y, frame.z], axes=(1,0))
    x_T = np.array([frame.x, frame.y, frame.z])
  for j, index in enumerate(point_index):
    if len(rotation) <= j:
      rotation.append([])
    if with_translation:
      # rotation[j].append(
      #   [x_T[index][0] + center[0], x_T[index][1] + center[1], x_T[index][2]])
      rotation[j].append(
        [frame.x[index], frame.y[index], frame.z[index]])
    else:
      rotation[j].append(
        [x_T[0][index] - center[0], x_T[1][index] - center[1], x_T[2][index]])
      # rotation[j].append(
      #   [frame.x[index] - center[0], frame.y[index] - center[1], frame.z[index]])

flip_x = False
for i in range(len(rotation)):
  x, y, z = np.transpose(rotation[i]) * 0.065
  x -= ((np.max(x) + np.min(x)) / 2 )
  z -= np.min(z) - 1
  y -= ((np.max(y) + np.min(y)) / 2 )
  plt.xlim(-5, 5)
  plt.ylim(-5, 5)
  N = len(x)
  for j in range(N - 1):
    if flip_x:
      ax.plot(-x[j:j + 2], y[j:j + 2], z[j:j + 2],
              color=plt.cm.jet(j / N), alpha=1)
      ax.plot(-x[j:j + 2], y[j:j + 2], 0,
              color=plt.cm.gray(j / N), alpha=1)
    else:
      ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
              color=plt.cm.jet(j / N), alpha=1)
      ax.plot(x[j:j + 2], y[j:j + 2], 0,
              color=plt.cm.gray(j / N), alpha=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# plt.show()

fig = plt.gcf()
fig.set_dpi(300)
plt.show()

