import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# load the pkl from the your disk
# data = 'T5P5_4'
data = 'three_before_division'
with open(f'./cell_class/{data}.pkl', 'rb') as f:
  cell = pickle.load(f)

rotation = [0]
local_r = 0

# Following codes show the line chart of rotation
# predefine a rotation_axis
#TODO: it may appreciated that we draw angle linechart in three views, e.g. x, y, z :)
rotation_axis = np.array([0, 0, 1])
for i, frame in enumerate(cell.frames):
  # if i % 10 != 0: continue
  rotvec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True)
  if rotation_axis @ rotvec > 0:
    local_r += np.linalg.norm(
      R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
  else:
    local_r += -np.linalg.norm(
      R.from_matrix(frame.locale_r).as_rotvec(degrees=True))
  rotation.append(local_r)
x_axis = np.arange(len(rotation))
plt.plot(x_axis, rotation)
plt.show()

# following codes are to visualize point traces.
# You specify the point index you would like to visualize
point_index = [0]
# Set to Ture if you want to draw the rotation axis trace instead of the point
vis_rotation_axis = False
# Set to Ture if you want to draw trace with translation information
with_translation = False
if vis_rotation_axis:
  point_index = [0, 1, 2]

matplotlib.use('tkagg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line = []
rotation = []
for i, frame in enumerate(cell.frames):
  # if i % 10 != 0: continue
  center = frame.center
  if vis_rotation_axis:
    x_T = np.transpose(frame.r * frame.radius * 2)
  else:
    x_T = np.transpose([frame.x, frame.y, frame.z], axes=(1,0))
  for j, index in enumerate(point_index):
    if len(rotation) <= j:
      rotation.append([])
    if with_translation:
      rotation[j].append(
        [x_T[index][0] + center[0], x_T[index][1] + center[1], x_T[index][2]])
    else:
      rotation[j].append(
        [x_T[index][0], x_T[index][1], x_T[index][2]])

flip_x = False
for i in range(len(rotation)):
  x, y, z = np.transpose(rotation[i]) * 0.065
  N = len(x)
  for j in range(N - 1):
    if flip_x:
      ax.plot(-x[j:j + 2], y[j:j + 2], z[j:j + 2],
              color=plt.cm.jet(j / N), alpha=1)
    else:
      ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
              color=plt.cm.jet(j / N), alpha=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

