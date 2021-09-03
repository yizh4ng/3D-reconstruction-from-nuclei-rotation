import json
import pickle
import pandas as pd
from data_cleasing import read_x_y_from_pkl
import numpy as np
from lambo.gui.vinci.vinci import DaVinci
from visualize_xyz import visualize_xyz

def classify_by_rotation_direction(x, y):
  displacement = (np.array((x[0], y[0])) - np.array((x[1], y[1]))).T
  max_displacement = displacement[np.linalg.norm(displacement, axis=1).argmax()]
  print(max_displacement)
  front = max_displacement @  (np.array((x[0], y[0])) - np.array((x[1], y[1])))
  # print(displacement.T @ (np.array((x[0], y[0])) - np.array((x[1], y[1]))))
  prediction = np.where(front > 0, -1,1)
  return prediction

def predict_origin(x,y):
  x_ = np.average(x)
  y_ = np.average(y)
  return  x_, y_

def predict_radius(x,y, origin):
  x_, y_ = origin

  return np.max(np.sqrt((x - x_)**2 + (y - y_)**2)) + 0.1

def predict_z(radius, x,y):

  x_, y_ = predict_origin(x[0], y[0])
  radius = predict_radius(x[0],y[0],(x_, y_)) + 1
  # front = classify_by_rotation_direction(x, y)
  depth = np.sqrt(radius**2 - (x[0] - x_)**2 - (y[0] - y_)**2)
  return depth

def predict_z(df:pd.DataFrame):
  for i in range(int(df['frame'].max()) + 1):
    frame:pd.DataFrame = df[df['frame'] == i]
    x = frame['x'].tolist()
    y = frame['y'].tolist()
    x_, y_ = predict_origin(x, y)
    radius = predict_radius(x, y, predict_origin(x, y))
    assert all(radius**2 - (np.array(x) - x_)**2 - (np.array(y) - y_)**2 >= 0)
    df['z'].loc[frame.index.tolist()] = \
      np.sqrt(radius**2 - (np.array(x) - x_)**2 - (np.array(y) - y_)**2).\
        tolist()
if __name__ == '__main__':

  f = open('data.pkl', 'rb')
  df:pd.DataFrame = pickle.load(f)
  df = df[ df['z'] > 0 ]
  df.drop('z', axis=1,inplace=True)
  df['z'] = 0
  predict_z(df)

  print(df.head(40))

  vis = visualize_xyz()
  x_g, y_g, z_g = vis.read_df(df)
  max_particle = 100
  max_frame = 100
  vis.vis_animation(x_g, y_g, z_g)
  vis.show()
