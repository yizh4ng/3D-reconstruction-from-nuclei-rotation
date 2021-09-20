# Build image matrix for further SVD
import json
import numpy as np
def build_matrix(x:list, y:list):
  x = np.array(x)
  y = np.array(y)
  return np.concatenate((x,y), axis=0)

if __name__ == '__main__':
  x = json.load(open('./ground_truth/x.json'))
  y = json.load(open('./ground_truth/y.json'))
  print(build_matrix(x,y).shape)