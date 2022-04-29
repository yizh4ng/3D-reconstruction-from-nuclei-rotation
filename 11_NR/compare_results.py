import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import numpy as np
from lambo.gui.vinci.vinci import DaVinci

def nearest_length(x, cor_list):
  spatial_difference = np.expand_dims(np.array(x), 0) - np.array(cor_list)
  spatial_difference_norm = np.linalg.norm(spatial_difference, axis=-1)
  return np.min(spatial_difference_norm)

def furthest_length(x, cor_list):
  spatial_difference = np.expand_dims(np.array(x), 0) - np.array(cor_list)
  spatial_difference_norm = np.linalg.norm(spatial_difference, axis=-1)
  return np.max(spatial_difference_norm)


class ReconstructionVisualizer(DaVinci):
  def __init__(self):
    super(ReconstructionVisualizer, self).__init__()

  def compare_ground_truth(self, x):

    ax = plt.axes(projection='3d')
    gt = x[0]
    results = x[1]
    loss_index = x[2]
    ax.scatter3D(*gt, c='blue')

    for i in range(len(loss_index)):
      x, y ,z = results[0][i],results[1][i],results[2][i],
      if loss_index[i] == 0:
        ax.scatter3D(x, y, z, c='orange')
      else:
        ax.scatter3D(x, y, z, c='red')

  def compare_trace(self, x, index, ax):
    gt_all = []
    reconstruct_all = []
    for ob in self.objects:
      gt_all.append(ob[0])
      reconstruct_all.append(ob[1])

    for point_index in index:
      gt = np.array(gt_all)[:, :, point_index]
      reconstruct = np.array(reconstruct_all)[:, :, point_index]

      x, y, z = np.transpose(gt)
      N = len(x)
      for j in range(N - 1):
        ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet(j / N))
      # ax.plot3D(*np.transpose(rotation[i]), c=color_dict[i])
      x, y, z = np.transpose(reconstruct)
      # N = len(x)
      for j in range(N - 1):
        ax.plot(x[j:j + 2], y[j:j + 2], z[j:j + 2],
                color=plt.cm.jet(j / N))

  def calculate_error(self, ):
    error = 0
    visible_error = 0
    lost_error = 0
    individual_error = np.zeros(len(self.objects[0][2]))
    for _, ob in enumerate(self.objects):
      gt = np.array(ob[0])

      norm_factor = furthest_length([0, 0, 0], np.array(gt).T)
      results = ob[1]
      loss_index = ob[2]
      for i in range(len(loss_index)):
        x, y ,z = results[0][i],results[1][i],results[2][i],
        if loss_index[i] == 0:
          visible_error += (nearest_length([x, y, z], np.array(gt).T) / norm_factor)**2 / (len(loss_index) - np.sum(loss_index)) / len(self.objects)
        else:
          lost_error += (nearest_length([x, y, z], np.array(gt).T) / norm_factor) ** 2 / np.sum(loss_index) / len(self.objects)

        error += (nearest_length([x,y,z], np.array(gt).T) /norm_factor)**2 / len(loss_index) / len(self.objects)
        individual_error[i] += (nearest_length([x,y,z], np.array(gt).T) /norm_factor) **2 / len(self.objects)
    print(len(self.objects[0][2]))
    plt.xticks(np.array(range(len(individual_error))) + 1)
    plt.bar(np.array(range(len(individual_error))) + 1, individual_error)
    plt.show()
    return error, visible_error, lost_error, individual_error



if __name__ == '__main__':
  alpha = 0.5
  frame = 4
  error = 0
  visible_error = 0
  missing_error = 0
  da = ReconstructionVisualizer()
  df_all = pd.read_pickle(f'./pkl/confocal_all.pkl')
  df = pd.read_pickle(f'./pkl/{alpha}_height.pkl')
  with open(f'./results/{alpha}_height/cell.pkl', 'rb') as f:
    reconstruct = pickle.load(f)

  keep_list = []
  for i, row in df_all.iterrows():
    if row['particle'] in set(df['particle'].tolist()):
      keep_list.append(True)
    else:
      keep_list.append(False)
  df_all = df_all[keep_list]

  da.df = df_all
  for frame in range(11):
    df_all_ = df_all[df_all['frame'] == frame]
    cor_list = np.array([df_all_['x'], df_all_['y'], df_all_['z'] - np.max(df_all_['z'])]).T
    object = [[df_all_['x'], df_all_['y'], df_all_['z'] - np.max(df_all_['z'])]]
    max_height = np.max(reconstruct['z'])
    x, y, z = reconstruct['x'][frame], reconstruct['y'][frame], \
              reconstruct['z'][frame] - max_height
    object.append([x,y,z])
    object.append(reconstruct['missing'][frame])

    da.objects.append(object)
  da.add_plotter(da.compare_ground_truth)
  da.add_plotter(lambda :da.compare_trace(da, [0, 7, 8], da.axes3d))
  print(da.calculate_error())
  da.show()


