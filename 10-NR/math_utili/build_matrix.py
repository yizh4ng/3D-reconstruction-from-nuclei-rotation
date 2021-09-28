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
  y_notNAN = np.argwhere(~np.isnan(data.T[y]))
  x3 = np.sum(np.array(list(itertools.combinations(x_notNAN, 3))),axis=2)
  y3 = np.sum(np.array(list(itertools.combinations(y_notNAN, 3))),axis=2)
  comb = np.array(list(itertools.product(y3, x3)))
  # print(comb.shape)
  return comb

def data_cleasing(x:np.array):
  x = np.delete(x, np.where(np.sum(~np.isnan(x), axis=1) < 3), axis=0)
  x = np.delete(x.T, np.where(np.sum(~np.isnan(x.T), axis=1) < 3), axis=0).T
  return x

def find_NAN_location(x:np.array):
  return np.argwhere(np.isnan(x))

def build_matrix_from_df(df:pd.DataFrame):
  x = []
  y = []
  for i in range(int(df['frame'].max()) + 1):
    x_ = np.array([])
    y_ = np.array([])
    for j in range(int(df['particle'].max()) + 1):
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
def find_submatrix(data, NAN_location):
  #F_P_comb = np.random.shuffle(F_P_comb)
  sub_matrix = []
  for NAN_loc in NAN_location:
    # print(NAN_loc)
    for F_P_c in get_notNAN_row_and_col(data, NAN_loc[0], NAN_loc[1]):
      F = np.concatenate((F_P_c[0], [NAN_loc[0]]))
      P = np.concatenate((F_P_c[1], [NAN_loc[1]]))
      Visible_points = list(itertools.product(F, P))
      Visible_points.pop()
      #print(F, P, len(Visible_points))
      #print(NAN_location)

      Visible_points_index = 100 * np.array(Visible_points)[:,0] \
                             + np.array(Visible_points)[:,1]
      NAN_loc_index = 100 * np.array(NAN_location)[:, 0] \
                             + np.array(NAN_location)[:, 1]
      #print(~np.isin(Visible_points_index, NAN_loc_index))
      #print(np.isin(np.array(Visible_points)[:,0], np.array(NAN_location)))
      if ((~np.isin(Visible_points_index, NAN_loc_index)).all()):
          #and (~np.isin(np.array(NAN_loc[0]), F_P_c[0])) \
          #and (~np.isin(np.array(NAN_loc[1]), F_P_c[1])):
        print(f'{NAN_loc} get matrix {F_P_c}')
        sub_matrix.append(F_P_c)
        break
  assert len(sub_matrix) == len(NAN_location)
  return sub_matrix



if __name__ == '__main__':
  x = json.load(open('../ground_truth/x.json'))
  y = json.load(open('../ground_truth/y.json'))
  df = pickle.load(open("../data_real.pkl", 'rb'))
  x, y = build_matrix_from_df(df)
  # print(build_matrix(x,y).shape)