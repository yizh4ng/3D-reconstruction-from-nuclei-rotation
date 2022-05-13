from back_projection import back_projection
from crop_roi import crop_roi
import pickle
import os
import numpy as np
import pims
from equalize_rotation import equalize_rotation
import matplotlib


if __name__ == '__main__':
  data = 'adam'
  # data = 'three_before_division'
  result_path = f"C:/Users/Administrator/Desktop/lambai/11_NR/cell_class/{data}.pkl"
  file_path = f"C:/Users/Administrator/Desktop/lambai/11_NR/data/{data}_bf.tif"
  sampling = 3
  if not os.path.exists(file_path):
    raise FileNotFoundError('!! File `{}` not found.'.format(file_path))
  frames = pims.open(file_path)
  # frames = frames[2:17]
  with open(result_path, 'rb') as f:
    rotating_cell = pickle.load(f)

  radius = rotating_cell.radius * 1.2
  # rotation = dict['r']
  center = rotating_cell.center

  cropped_frames = crop_roi(frames, radius, center)

  vol_size = int(radius * 3)
  vol = np.zeros((vol_size, vol_size, vol_size))

  vol = vol[::sampling,::sampling,::sampling]
  rotation_list = []
  for i, frame in enumerate(rotating_cell.frames):
    rotation_list.append(frame.r)

  rotation_list, index_list = equalize_rotation(rotation_list)

  for i, rotation in enumerate(rotation_list):
    if i % 5 != 0:continue
    # if i == 0:continue
    print(i)
    vol = back_projection(vol, cropped_frames[index_list[i]][::sampling, ::sampling], rotation, crop_circle=False)

  # vol = vol[int(0.5 * vol_size):-int(0.5 * vol_size),
  #       int(0.5 * vol_size):-int(0.5 * vol_size),
  #       int(0.5 * vol_size):-int(0.5 * vol_size),]
  vol -= np.percentile(vol, 95)
  vol[vol < 0 ] = 0
  X_len, Y_len, Z_len = vol.shape
  print(vol.shape)
  X, Y, Z = np.mgrid[0:X_len, 0:Y_len, 0:Z_len]

  import plotly.graph_objs as go

  from lambo.gui.vinci.vinci import DaVinci
  da = DaVinci()
  da.objects = vol
  da.add_plotter(da.imshow)
  da.show()
  X = X.flatten()
  Y = Y.flatten()
  Z = Z.flatten()
  vol = vol.flatten()
  print(vol)
  fig = go.Figure(data=go.Volume(
    x=X,
    y=Y,
    z=Z,
    value=vol,
    isomin=0.1,
    # isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=5, # needs to be a large number for good volume rendering
  ))
  fig.show()

