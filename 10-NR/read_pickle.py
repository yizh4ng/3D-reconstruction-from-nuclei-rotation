import json
import numpy as np
import pandas as pd
import numpy as n

def read_to_xyz(file:str):
  x, y, z = [], [], []
  data =  pd.read_pickle(file)
  for i in range(0, int(data['frame'].max())+1):
    x.append(data[data['frame'] == i]['x'].values)
    y.append(data[data['frame'] == i]['y'].values)
    z.append(data[data['frame'] == i]['z'].values)
  return x, y, z
if __name__ == '__main__':
  x, y, z = read_to_xyz('data_Dai.pkl')
  print(np.array(y).shape)