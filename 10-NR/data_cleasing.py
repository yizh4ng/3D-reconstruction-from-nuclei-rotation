import json
import numpy as np

import pandas as pd

def get_num_anchors(frame,frame_index= 0):
  return len(frame[frame['frame']==frame_index])

def remove_anchors(frame, max_particle):
  return frame[frame['particle']<max_particle]

def remove_unlink(frame, max_particle):
  particle_to_remove = []
  for frame_index in range(frame['frame'].max()):
    for particle_index in range(max_particle + 1):
      if particle_index not in frame[frame['frame'] == frame_index]['particle'].tolist()\
          and particle_index not in particle_to_remove:
        particle_to_remove.append(particle_index)
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
  x, y = read_x_y_from_pkl('data_3.pkl')



