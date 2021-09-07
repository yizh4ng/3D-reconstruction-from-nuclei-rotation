import json
import pickle

import numpy as np

import pandas as pd

def get_particle_list(dataframe):
  return dataframe[dataframe['frame'] == 0 ]['particle'].tolist()

def remove_anchors(frame:pd.DataFrame, particles: list):
  return frame[frame['particle'].isin(particles)]

def remove_unlink(frame:pd.DataFrame):
  frame = remove_anchors(frame, get_particle_list(frame))
  particle_to_remove = []
  appear_time = frame['frame'].max() + 1
  for p in get_particle_list(frame):
    if(frame[frame['particle'] == p].shape[0] != appear_time):
      particle_to_remove.append(p)

  frame = frame[~frame['particle'].isin(particle_to_remove)]
  return frame

def get_anchors(frame):
  return frame[frame['frame'] == 0]['particle'].tolist()

def read_x_y_from_pkl(path, max_frame = 0, max_particle = 0):
  data = pd.read_pickle(path)
  data = data.filter(items=['x', 'y', 'particle', 'frame'])
  num_anchors = get_num_anchors(data, frame_index=0)
  data = remove_anchors(data, num_anchors)
  data = remove_unlink(data, num_anchors)
  if max_frame != 0:
    data = data[data['frame'] < max_frame]
  if max_particle != 0:
    data = data[data['particle'] < max_particle]
  print(data)
  data_by_frames = []

  for i in range(data['frame'].max() + 1):
    data_by_frames.append(data[data['frame'] == i])
  x = []
  y = []
  for frame in data_by_frames:
    frame = frame.sort_values('particle')
    xx = frame['x'].tolist()
    yy = frame['y'].tolist()
    x.append(xx)
    y.append(yy)

  return x,y

def read(path):
  x = json.load(open(f'./{path}/x.json'))
  y = json.load(open(f'./{path}/y.json'))
  z = json.load(open(f'./{path}/z.json'))
  return x, y, z



if __name__ == '__main__':
  # x, y = read_x_y_from_pkl('data.pkl')
  f = open('data.pkl', 'rb')
  df = pickle.load(f)
  df = df[df['z'] > 0]
  print(get_particle_list(df))
  # print(remove_anchors(dataframe, paticles))
  print(get_particle_list(remove_unlink(df)))


