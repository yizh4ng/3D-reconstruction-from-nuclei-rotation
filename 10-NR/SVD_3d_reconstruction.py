import json
import pickle

import math_utili.solve
import math_utili.SVD
import numpy as np
import numpy.linalg as LA
from scipy.optimize import fsolve,leastsq

from visualize_xyz import visualize_xyz
from lambo.gui.vinci.vinci import DaVinci
def SV_D(M:np.array):
  return np.linalg.svd(M)

def select_3_most_SV(P, E, Q):
  return P[:,0:3], np.diag(E[0:3]), Q[0:3, :]

def SVD_reconstruct(M:np.array):
  P, E, Q = SV_D(M)
  P, E, Q = select_3_most_SV(P, E, Q)
  sol = math_utili.solve.solve_x_y(x, y, P, E, Q)
  # print(sol)
  P = np.matmul(P, np.sqrt(E))
  Q = np.matmul(np.sqrt(E), Q)
  S = np.expand_dims(np.matmul(LA.inv(sol), Q), axis=0)
  R = np.expand_dims(np.matmul(P, sol), axis=0)
  return R, S

def equations(i, P):
  u,v,s = P[0], P[1], P[2]
  equations = []
  for k in range(3):
    equations.append(np.dot(i[:3], s[k]) - u[k])
  for k in range(3):
    equations.append(np.dot(i[3:], s[k]) - v[k])
  equations.append(np.linalg.norm(i[:3]) - 1)
  equations.append(np.linalg.norm(i[3:]) - 1)
  equations.append(np.dot(i[:3],i[3:]))
  return equations

def reconstruct_missing(x, y, nan_loc, submatrix):
  x_f = x[np.concatenate(([nan_loc[0]], submatrix[0]))]
  x_f = x_f.T[np.concatenate(([nan_loc[1]], submatrix[1]))].T
  y_f = y[np.concatenate(([nan_loc[0]], submatrix[0]))]
  y_f = y_f.T[np.concatenate(([nan_loc[1]], submatrix[1]))].T

  M = np.concatenate((x_f[1:,:], y_f[1:,:]), axis=0)
  t = np.sum(M.T[1:].T,axis=1)/3
  M_prime = M - np.outer(t, np.ones(4))
  R, S = SVD_reconstruct(M_prime)
  i = np.zeros(6)
  P1 =[x_f[0][1:] - np.sum(x_f[0][1:]/3), y_f[0][1:] - np.sum(y_f[0][1:]/3), S[0].T[1:]]
  solution = leastsq(equations, i, args=P1)[0]
  i4 = solution[0:3]
  j4 = solution[3:]
  # print(np.linalg.norm(i4), np.linalg.norm(j4), np.dot(i4, j4))
  R = R[0]
  R = np.insert(R, 0, i4, axis=0)
  R = np.insert(R, 4, j4, axis=0)
  M = np.concatenate((x_f, y_f), axis=0)
  a = np.dot((M.T[1:].T-np.matmul(R, S[0]).T[1:].T), np.array([1,1,1]))/3
  x[nan_loc[0]][nan_loc[1]] = np.dot(i4, S[0].T[0]) + a[0]
  y[nan_loc[0]][nan_loc[1]] = np.dot(j4, S[0].T[0]) + a[4]



if __name__ == '__main__':
  '''x = json.load(open('./ground_truth/x.json'))
  y = json.load(open('./ground_truth/y.json'))
  M = math_utili.build_matrix.build_matrix(x,y)
  s=SVD_reconstruct(M)
  vis = visualize_xyz()
  vis.objects = np.array(s)
  vis.add_plotter(vis.draw_3d)
  vis.show()'''
  df = pickle.load(open("./data_real.pkl", 'rb'))
  df1 = pickle.load(open("./data.pkl", 'rb'))
  x1, y1 = math_utili.build_matrix.build_matrix_from_df(df1)
  x, y = math_utili.build_matrix.build_matrix_from_df(df)
  x = math_utili.build_matrix.data_cleasing(x)
  y = math_utili.build_matrix.data_cleasing(y)
  nan_location = math_utili.build_matrix.find_NAN_location(x)
  # F_P_comb = math_utili.build_matrix.all_3X3_FP_comb(x.shape[0], x.shape[1])
  sub_matrix = math_utili.build_matrix.find_submatrix(x, nan_location)
  #reconstruct_missing(x, y, [0, 9], [[7,10,11], [1,2,3]])
  for i in range(len(nan_location)):
    reconstruct_missing(x, y, nan_location[i], sub_matrix[i])
  #a= math_utili.build_matrix.get_notNAN_row_and_col(x,0,9)
  print("finish")
  # print(math_utili.build_matrix.find_submatrix(NAN_location))
  # print(F_P_comb.shape)
