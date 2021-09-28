import numpy as np
import json
from .build_matrix import build_matrix
from scipy import linalg
def SV_D(M:np.array):
  return np.linalg.svd(M)

def select_3_most_SV(P, E, Q):
  return P[:,0:3], np.diag(E[0:3]), Q[0:3, :]

if __name__ == '__main__':
  x = json.load(open('../ground_truth/x.json'))
  y = json.load(open('../ground_truth/y.json'))
  P, E, Q = select_3_most_SV(*SV_D(build_matrix(x,y)))
  print(P,E,Q)