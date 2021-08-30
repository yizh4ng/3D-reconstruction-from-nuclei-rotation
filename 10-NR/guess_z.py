import json

from data_cleasing import read_x_y_from_pkl
import numpy as np
from lambo.gui.vinci.vinci import DaVinci

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

  return np.max(np.sqrt((x - x_)**2 + (y - y_)**2))

def predict_z(x,y):

  x_, y_ = predict_origin(x[0], y[0])
  radius = predict_radius(x[0],y[0],(x_, y_)) + 0.01
  front = classify_by_rotation_direction(x, y)
  depth = np.sqrt(radius**2 - (x[0] - x_)**2 - (y[0] - y_)**2)
  return depth * front


if __name__ == '__main__':
  # x, y = read_x_y_from_pkl('data_3.pkl', max_particle=20, max_frame=3)
  x = json.load(open('./ground_true/x.json'))
  y = json.load(open('./ground_true/y.json'))

  #prediction = classify_by_rotation_direction(x, y)
  z = predict_z(x,y)
  print(z)
  da = DaVinci()
  def vis():
    da.axes3d.view_init(elev=90, azim=0)
    da.axes3d.plot(x[0],y[0],z,'ro', markersize=2)


  da.add_plotter(vis)
  da.show()