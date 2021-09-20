from .SVD import select_3_most_SV, SV_D
import json
from .build_matrix import build_matrix
import numpy as np
import numpy.linalg as LA
from scipy.optimize import fsolve,leastsq

def equations(v:np.array, P):
  l = int(len(P)/2)
  equations = []
  v_ = np.reshape(v, (3,3))
  '''equations.append(LA.norm(np.matmul(np.array([1, 0, 0]), v_)) - 1)
  for row in P[1:l]:
    equations.append(LA.norm(np.matmul(row, v_))-1)
  equations.append(LA.norm(np.matmul(np.array([0, 1, 0]), v_)) - 1)
  for row in P[l+1:]:
    equations.append(LA.norm(np.matmul(row, v_))-1)'''
  for row in P:
    equations.append(LA.norm(np.matmul(row, v_)) - 1)

  for i in range(l):
    equations.append(np.dot(np.matmul(P[i], v_),
                            np.matmul(P[i + l], v_)))

  return equations

def solve_x_y(x,y, P, E):
  P, E, Q = select_3_most_SV(*SV_D(build_matrix(x, y)))
  x = np.zeros(9)
  sol = leastsq(equations, x, args=np.matmul(P, np.sqrt(E)))
                #args=P)
  sol = np.reshape(sol[0], (3,3))
  return sol


if __name__ == '__main__':
  x = json.load(open('../ground_truth/x.json'))
  y = json.load(open('../ground_truth/y.json'))
  sol = solve_x_y(x,y, P, E)
  print(sol)