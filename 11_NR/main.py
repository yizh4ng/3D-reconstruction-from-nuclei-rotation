import pandas as pd
import numpy as np

from rotating_cell import Rotating_Cell
from visualize_center import Cell_Visualizer



if __name__ == '__main__':
  df = pd.read_pickle('./data_real_all.pkl')
  # df = df[(df['frame'] >= 75) & (df['frame'] <= 114) & (
  #     df['particle'] != 14)  # 65 ~ 109
  #         & (df['particle'] != 47)]
  # df = df.sort_values(['frame', 'particle'], ascending=[True, True])
  cell = Rotating_Cell(df)
  cell.run()


  x = np.expand_dims(cell.x, axis=1)
  y = np.expand_dims(cell.y, axis=1)
  xy = np.concatenate((x, y), axis = 1)
  center_ = np.expand_dims(cell.center, axis = -1)
  xy_with_center = np.concatenate((center_, xy), axis = -1)
  points = cell.frames[0].points
  cv = Cell_Visualizer(cell)
  # cv.objects = xy_with_center
  cv.add_plotter(cv.draw_frame_2d)
  cv.add_plotter(cv.draw_frame_3d)
  cv.add_plotter(cv.draw_rotation)
  cv.show()
