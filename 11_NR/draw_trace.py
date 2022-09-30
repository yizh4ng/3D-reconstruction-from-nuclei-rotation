import os, pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



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

path = r'Z:\Data to ZHANG Yi\figure_modification\Fig 4 19T2 Position 1 pre-mitosis rotation'
# data = 'Yijin Rotated HEK_20220512P9_Cell1_ICC_RAW_ch00_3d.pkl'
data = 'three_before_division.pkl'
with open(os.path.join(path, data), 'rb') as f:
  cell = pickle.load(f)

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
      rotation[j].append(
        [frame.x[index], frame.y[index], frame.z[index]])
    else:
      rotation[j].append(
        [x_T[0][index] - center[0], x_T[1][index] - center[1], x_T[2][index]])

flip_x = False
for i in range(len(rotation)):
  x, y, z = np.transpose(rotation[i]) * 0.065
  x -= ((np.max(x) + np.min(x)) / 2 )
  # z -= np.min(z) - 1
  y -= ((np.max(y) + np.min(y)) / 2 )
  # plt.xlim(-5, 5)
  # plt.ylim(-5, 5)
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
# for ii in range(0,360, 10):
#         ax.view_init(elev=10., azim=ii)
#         plt.savefig(rf"{path}\traces\movie%d.png" % ii)
# plt.savefig(rf"{path}\{data}_trace.png", dpi=300)
fig = plt.gcf()
fig.set_dpi(300)
plt.show()

