import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
  f_x = np.exp(x) / np.sum(np.exp(x))
  return f_x

def entropy(x):
  return -np.sum(x * np.log(x))

def get_axis_eigen(cell, axis=np.array([0, 0, 1]), verbose=False):
  rotation = []
  for i, frame in enumerate(cell.frames):
    # if i % 3 == 0: continue
    rotvec = R.from_matrix(frame.locale_r).as_rotvec(degrees=True)
    if np.linalg.norm(rotvec) ==0:
      continue
    # rotation_axis = rotvec / np.linalg.norm(rotvec)
    rotation_axis = rotvec
    # if rotation_axis @ axis < 0:
    #   rotation_axis = -rotation_axis
    rotation.append(rotation_axis)

  rotation = np.transpose(np.array(rotation))
  if np.linalg.matrix_rank(rotation) < 3:
    print('SVD does not converge.')
    return None
  P, E, Q = np.linalg.svd(rotation)

  E = E / np.sum(E)
  eigen = 0
  for i, rot_axis in enumerate(np.transpose(P)):
    # if i > 0: break
    eigen += np.abs(rot_axis @ axis *  E[i])
    # eigen += np.abs(rot_axis @ axis * np.tan(np.pi / 2 * E[i]))
    #* 1 / entropy(E))
  if verbose:
    print('Eignvectors: \n', np.transpose(P))
    print('Eignvalues: ', E)
    print(eigen)
    # print('Entropy: ', entropy(E))
  return eigen

# load the pkl from the your disk
# data = 'T5P5_4'
# data = 'adam'
# data = 'three_before_division'
# data = 'unsyn_cos_1'
# with open(f'./cell_class/{data}.pkl', 'rb') as f:
#   cell = pickle.load(f)

if __name__ == '__main__':
  SAVE = False
  save_path = r'Z:\Data to ZHANG Yi\figure_modification\Fig 6 HEK vs COS7 quantification'
  from roma import finder
  import matplotlib.pyplot as plt
  # roots = [r'Z:\Data to ZHANG Yi\20 Rotated HEK 293T for quantification',
  #         r'Z:\Data to ZHANG Yi\20 Rotated COS-7 for quantification']
  # root = r'Z:\Data to ZHANG Yi\20 Rotated COS-7 for quantification'
  # roots = [r'./cell_class']
  roots = [r"Z:\Data to ZHANG Yi\20 Rotated HEK 293T for quantification\Yijin Rotated HEK_20220512P9"]

  def get_all_eigen(path):
    pkl_paths = finder.walk(path, pattern='*3d.pkl', recursive=True)
    print(len(pkl_paths))
    eigens = []
    for path in pkl_paths:
      with open(path, 'rb') as f:
        pkl = pickle.load(f)
      eigen = get_axis_eigen(pkl)
      if eigen is not None:
        eigens.append(get_axis_eigen(pkl, axis=np.array([0,0,1]), verbose=True))
        print(path)
        if 'HEK' in path and eigen > 0.5:
          print(f'Abnormal Hek{eigen}:' + path)
        if 'COS' in path and eigen < 0.5:
          print(f'Abnormal cos7{eigen}:' + path)
    return eigens

  for i, root in enumerate(roots):
    print(root)
    eigens = get_all_eigen(root)
    print(eigens)
    print(np.mean(eigens), np.std(eigens))

    eigens = np.array(eigens)
    bins = np.linspace(0, 1, 100)
    plt.hist(eigens, bins=bins,alpha=0.5)
    if SAVE:
      import pandas as pd
      import os
      if 'HEK' in roots[i]:
        pd.DataFrame(eigens).to_excel(os.path.join(save_path, 'hek.xlsx'))
      if 'COS' in roots[i]:
        pd.DataFrame(eigens).to_excel(os.path.join(save_path, 'cos.xlsx'))
  plt.show()