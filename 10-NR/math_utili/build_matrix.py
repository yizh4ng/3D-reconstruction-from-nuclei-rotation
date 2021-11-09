# Build image matrix for further SVD
import itertools
import pickle

import pandas as pd
import json
import numpy as np
def build_matrix(x:list, y:list):
  x = np.array(x)
  y = np.array(y)
  return np.concatenate((x,y), axis=0)

def get_notNAN_row_and_col(data, x, y):
  x_notNAN = np.argwhere(~np.isnan(data[x]))
  x_distance = np.abs(np.sum(x_notNAN, axis=-1) - y)
  x_notNAN = x_notNAN[np.argsort(x_distance)]
  y_notNAN = np.argwhere(~np.isnan(data.T[y]))
  y_distance = np.abs(np.sum(y_notNAN, axis=-1) - x)
  y_notNAN = y_notNAN[np.argsort(y_distance)]
  x3 = np.sum(np.array(list(itertools.combinations(x_notNAN, 3))),axis=2)
  y3 = np.sum(np.array(list(itertools.combinations(y_notNAN, 3))),axis=2)
  comb = np.array(list(itertools.product(y3, x3)))
  # print(comb.shape)
  # print(f"[{x},{y}] gots {comb[:5]}")
  return comb

def data_cleasing(x:np.array):
  # for every frame, at least 3 particle
  x = np.delete(x, np.where(np.sum(~np.isnan(x), axis=1) < 3), axis=0)
  # for very particle, at least 3 frames
  x = np.delete(x.T, np.where(np.sum(~np.isnan(x.T), axis=1) < 5), axis=0).T
  return x

def find_NAN_location(x:np.array):
  return np.argwhere(np.isnan(x))

def get_frame_list(df:pd.DataFrame):
  ls = list(map(int, df['frame'].values))
  ls = list(set(ls))
  return sorted(ls)

def get_particle_list(df:pd.DataFrame):
  ls = list(map(int, df['particle'].values))
  ls = list(set(ls))
  return sorted(ls)

def build_matrix_from_df(df:pd.DataFrame):
  x = []
  y = []
  frame_list =get_frame_list(df)
  particle_list = get_particle_list(df)
  # for i in range(int(df['frame'].max()) + 1):
  for i in frame_list:
    x_ = np.array([])
    y_ = np.array([])
    for j in particle_list:
      row = df[(df['frame'] == i) & (df['particle'] == j)]
      if len(row) == 0:
        # print(f"particle{j} at frame {i} disappears")
        x_ = np.concatenate((x_, [np.NAN]))
        y_ = np.concatenate((y_, [np.NAN]))
      else:
        assert len(row) == 1
        row = row.iloc[0]
        # print(f"particle{j} at frame {i}at{row['x'], row['y']}")
        x_ = np.concatenate((x_, [row['x']]))
        y_ = np.concatenate((y_, [row['y']]))
    x.append(x_)
    y.append(y_)
  x = np.array(x)
  y = np.array(y)
  return x, y

def all_3X3_FP_comb(F, P):
  F_comb = np.array(
    list(itertools.combinations(np.linspace(0, F, F + 1).astype(int), 3)))
  P_comb = np.array(
    list(itertools.combinations(np.linspace(0, P, P + 1).astype(int), 3)))
  # print(F_comb.shape, P_comb.shape)
  F_P_comb = np.array(list(itertools.product(F_comb, P_comb)))
  return F_P_comb


# F_P_comb is N * 2 * 3, NAN_location is M * 2
# return M * 2 * 3
def find_submatrix(data:np.ndarray, NAN_location):
  #F_P_comb = np.random.shuffle(F_P_comb)
  sub_matrix = []
  for NAN_loc in NAN_location:
    # print(NAN_loc)
    find = False
    for F_P_c in get_notNAN_row_and_col(data, NAN_loc[0], NAN_loc[1]):
      F = np.concatenate((F_P_c[0], [NAN_loc[0]]))
      P = np.concatenate((F_P_c[1], [NAN_loc[1]]))
      Visible_points = list(itertools.product(F, P))
      Visible_points.pop()
      #print(F, P, len(Visible_points))
      #print(NAN_location)
      Visible_points_index = 1000 * np.array(Visible_points)[:,0] \
                             + np.array(Visible_points)[:,1]
      NAN_loc_index = 1000 * np.array(NAN_location)[:, 0] \
                             + np.array(NAN_location)[:, 1]
      #print(~np.isin(Visible_points_index, NAN_loc_index))
      #print(np.isin(np.array(Visible_points)[:,0], np.array(NAN_location)))
      if ((~np.isin(Visible_points_index, NAN_loc_index)).all()):
        print(f'{NAN_loc} get matrix {F_P_c}')
        sub_matrix.append(F_P_c)
        find = True
        break
    if not find:
      print(f'{NAN_loc} gots no submatrix !!!!!!!!!!!!!!!!!!!!!!!')
  assert len(sub_matrix) == len(NAN_location)
  return sub_matrix

def find_sub_matrix_by_one(data, NAN_loc, NAN_location):
  find = False
  # len = 9999
  for F_P_c in get_notNAN_row_and_col(data, NAN_loc[0], NAN_loc[1]):
    F = np.concatenate((F_P_c[0], [NAN_loc[0]]))
    P = np.concatenate((F_P_c[1], [NAN_loc[1]]))
    Visible_points = list(itertools.product(F, P))
    Visible_points.pop()
    # print(F, P, len(Visible_points))
    # print(NAN_location)
    Visible_points_index = 1000 * np.array(Visible_points)[:, 0] \
                           + np.array(Visible_points)[:, 1]
    NAN_loc_index = 1000 * np.array(NAN_location)[:, 0] \
                    + np.array(NAN_location)[:, 1]
    # print(~np.isin(Visible_points_index, NAN_loc_index))
    # print(np.isin(np.array(Visible_points)[:,0], np.array(NAN_location)))
    if ((~np.isin(Visible_points_index, NAN_loc_index)).all()):
      print(f'{NAN_loc} get matrix {F_P_c}')
      return F_P_c

  print(f'{NAN_loc} gets no matrix')
  return -1


def linear_padding(x:np.ndarray):
  for i in range(len(x)):
    for j in range(len(x[0])):
      if (np.isnan(x[i][j]) and (i + 1) < len(x) and (i - 1)> 0):
        x[i][j] = (x[i + 1][j] + x[i - 1][j])/2
  return x

if __name__ == '__main__':
  x = json.load(open('../ground_truth/x.json'))
  y = json.load(open('../ground_truth/y.json'))
  df = pickle.load(open("../data_real.pkl", 'rb'))
  x, y = build_matrix_from_df(df)
  # print(build_matrix(x,y).shape)