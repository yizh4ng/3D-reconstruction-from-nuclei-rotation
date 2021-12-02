import os
import pickle
import sys
sys.path.insert(0, "../../lambai")
sys.path.insert(0, '../roma')
import pandas as pd
import numpy as np
from roma import console

from rotating_cell import Rotating_Cell
from visualize_center import Cell_Visualizer
from optimize import Trainer

os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == '__main__':
  # df = pd.read_pickle('./data_real_all.pkl')
  data = 'adam'
  df = pd.read_pickle(f'./pkl/{data}.pkl')
  save = False
  steps = 1
  # df = df[(df['frame'] >= 75) & (df['frame'] <= 114) & (
  #     df['particle'] != 14) ] # 65 ~ 109
  #         & (df['particle'] != 47)]
  # df = df.sort_values(['frame', 'particle'], ascending=[True, True])
  # df = df[(df['frame'] >= 75) & (df['frame'] <= 400)]
  cell = Rotating_Cell(df, del_para=1.7, iteratively_op_radius=True,
                       iterative_times=9)
  cell.run()


  x = np.expand_dims(cell.x, axis=1)
  y = np.expand_dims(cell.y, axis=1)
  xy = np.concatenate((x, y), axis = 1)
  center_ = np.expand_dims(cell.center, axis = -1)
  xy_with_center = np.concatenate((center_, xy), axis = -1)

  if save:
    dict = {}
    dict['radius'] = cell.radius
    dict['radii'] = cell.radii
    dict['ellipse_direction'] = cell.ellipse_direciton
    x, y, z, center, r = [], [], [], [], []
    for f in cell.frames:
      x.append(f.x.tolist())
      y.append(f.y.tolist())
      z.append(f.z.tolist())
      center.append(f.center)
      r.append(f.r)
    dict['x'] = x
    dict['y'] = y
    dict['z'] = z
    dict['r'] = r
    dict['center'] = center
    with open(f'./results/{data}/cell.pkl', 'wb+') as f:
      pickle.dump(dict, f)
    with open(f'./cell_class/{data}.pkl', 'wb+') as f:
      pickle.dump(cell, f)
  cv = Cell_Visualizer(cell)
  # cv.train()
  cv.add_plotter(cv.draw_frame_2d)
  cv.add_plotter(cv.draw_frame_3d)
  cv.add_plotter(cv.draw_frame_3d_ellipse)
  cv.add_plotter(cv.draw_rotation)
  cv.show()
  # console.show_status('Start training...')
  # for i in range(steps):
  #   console.print_progress(i, total=steps)
  #   Trainer.train(cell)
  #
  # console.show_status('Finish training')