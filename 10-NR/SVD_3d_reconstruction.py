import json
import pickle
import copy
import math_utili.solve
import math_utili.SVD
import numpy as np
import numpy.linalg as LA
from scipy.optimize import fsolve,leastsq
import pandas as pd
from pandas import DataFrame
from visualize_xyz import visualize_xyz
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
  # t = np.sum(M.T[1:].T,axis=1)/ len(M[0])
  # M_prime = M - np.outer(t, np.ones(int(len(M)/2 + 1)))
  t_x = np.mean(x_f.T[:,1:], axis=1)
  t_y = np.mean(y_f.T[:,1:], axis=1)
  x_f_prime = x_f - np.outer(t_x, np.ones(int(len(M)/2 + 1)))
  y_f_prime = y_f - np.outer(t_y, np.ones(int(len(M)/2 + 1)))
  M_prime = np.concatenate((x_f_prime[1:,:], y_f_prime[1:, :]), axis=0)
  R, S = SVD_reconstruct(M_prime)
  i = np.zeros(6)
  P1 =[x_f[0][1:] - np.sum(x_f[0][1:]/3), y_f[0][1:] - np.sum(y_f[0][1:]/3), S[0].T[1:]]
  solution = leastsq(equations, i, args=P1)[0]
  i4 = solution[0:3]
  j4 = solution[3:]
  # print(np.linalg.norm(i4), np.linalg.norm(j4), np.dot(i4, j4))
  R = R[0]
  R = np.insert(R, 0, i4, axis=0)
  R = np.insert(R, int(len(R)/2) + 1, j4, axis=0)
  M = np.concatenate((x_f, y_f), axis=0)
  # a = np.dot((M.T[1:].T-np.matmul(R, S[0]).T[1:].T), np.array([1,1,1]))/3
  a = np.mean((M.T[1:].T - np.matmul(R, S[0]).T[1:].T),axis=-1)
  x[nan_loc[0]][nan_loc[1]] = np.dot(i4, S[0].T[0]) + a[0]
  y[nan_loc[0]][nan_loc[1]] = np.dot(j4, S[0].T[0]) + a[4]



if __name__ == '__main__':
  # x = json.load(open('./ground_truth/x.json'))
  # y = json.load(open('./ground_truth/y.json'))
  # M = math_utili.build_matrix.build_matrix(x,y)
  # r, s=SVD_reconstruct(M)
  # print(np.array(s).shape)
  # x = np.array(r[0][0])
  # y = np.array(r[0][30])
  # z = np.cross(x, y)
  # R = np.array([x , y, z])
  # print((R @ np.array(s[0])).shape)
  # vis = visualize_xyz()
  # vis.objects = np.array([R @ np.array(s[0])])
  # vis.add_plotter(vis.draw_3d)
  # vis.show()
  #
  # x = json.load(open('./ground_truth/x.json', 'rb'))
  # y = json.load(open('./ground_truth/y.json', 'rb'))
  # z = json.load(open('./ground_truth/z.json', 'rb'))
  #
  # Sn = np.array([x[0], y[0] ,z[0]])
  # print(np.linalg.norm(Sn - vis.objects[0]))

  # df = pickle.load(open("./data_real_all.pkl", 'rb'))
  df = pickle.load(open("../11_NR/pkl/step5.pkl", 'rb'))
  df = df[df['frame'] == df['frame']]
  print(df['frame'].max())
  # df:DataFrame= df[(df['frame'] >= 85) & (df['frame'] <= 105) & (df['particle'] != 14)# 65 ~ 109
  #                  & (df['particle'] != 47)]
  df = df.sort_values(['frame', 'particle'], ascending=[True, True])
  # df1 = pickle.load(open("./data.pkl", 'rb'))
  x, y = math_utili.build_matrix.build_matrix_from_df(df)
  x:np.ndarray = math_utili.build_matrix.data_cleasing(x)
  y:np.ndarray = math_utili.build_matrix.data_cleasing(y)
  x = math_utili.build_matrix.linear_padding(x)
  y = math_utili.build_matrix.linear_padding(y)
  # x = np.delete(x, [3,14], 1)
  # y = np.delete(y, [3,14], 1)
  #x = x[:10,:8]
  # y = y[:10,:8]
  x1 = copy.deepcopy(x)
  # nan_location = math_utili.build_matrix.find_NAN_location(x)
  # sub_matrix = math_utili.build_matrix.find_submatrix(x, nan_location)

  # reconstruct_missing(x,y,np.array([0, 10]), np.array([[21,22,23, 24, 25], [ 7, 6, 2,  1, 0]]))
  # # print(x[0,10])
  # for NAN_loc in nan_location:
  #   sub_matrix = math_utili.build_matrix.find_sub_matrix_by_one(x, NAN_loc, nan_location)
  #   if isinstance(sub_matrix, np.ndarray):
  #     reconstruct_missing(x, y, NAN_loc, sub_matrix)
      # nan_location = nan_location[1:]
      # print(len(nan_location))



  vis = visualize_xyz()
  vis.objects = []
  for i in range(len(x)):
    vis.objects.append([x[i],y[i]])
  vis.add_plotter(vis.draw_2d)
  vis.show()

  # with open('data/x.json', 'w') as f:
  #   json.dump(x.tolist(), f)
  # with open('data/y.json', 'w') as f:
  #   json.dump(y.tolist(), f)
  M = math_utili.build_matrix.build_matrix(x, y)
  # M  = np.delete()
  r, s = SVD_reconstruct(M)
  r = r[0]
  s = s[0]
  # s = s[:, s[-1, :] > 0 * np.max(s[-1, :])]
  R = []
  for i in range(int(len(r)/2)):
    perpen = np.cross(r[i], r[i + int(len(r)/ 2)])
    R_ = np.array([r[i], r[i + int(len(r) / 2)], perpen/np.linalg.norm(perpen)])
    R.append(R_)
  vis = visualize_xyz()
  for r in R:
    r_s = r @ s
    # r_s = r_s[:, r_s[-1, :] > 0 * np.max(r_s[-1, :])]
    vis.objects.append(r_s)
  # vis.objects = R @ s
  vis.add_plotter(vis.draw_2d)
  vis.add_plotter(vis.draw_3d)
  vis.show()

  save = True
  alpha = 0.5
  if save:
    df = pd.DataFrame(data={'frame':[], 'particle':[], 'x':[], 'y': []})
    max_height = np.max(np.array(vis.objects)[:,-1,:])
    for frame_id, ob in enumerate(vis.objects):
      for pt_id,pt in enumerate(ob.T):
        if pt[-1] > alpha * max_height:
          df = df.append({'frame': frame_id, 'particle': pt_id, 'x': pt[0], 'y': pt[1], 'z': pt[2]},
                    ignore_index=True)

    df.to_pickle(f'../11_NR/pkl/{alpha}_height.pkl')


