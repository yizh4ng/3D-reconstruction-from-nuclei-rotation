from lambo.analyzer.tracker import Tracker

import trackpy as tp
import pandas
import numpy as np
import matplotlib.pyplot as plt
class TrackerAlpha(Tracker):
  def __init__(self, df, **kwargs):
    super(TrackerAlpha, self).__init__(df, **kwargs)
    self.chosen_points = []
    self.mode = 'select'
    self.canvas.mpl_connect('button_press_event', self.select_point)
    self.state_machine.register_key_event('r', self.cancell_selected_points)
    self.state_machine.register_key_event('c', self.connect)
    self.state_machine.register_key_event('d', self.delete)
    self.state_machine.register_key_event('n', self.create)
    self.state_machine.register_key_event('x', self.delete_all)

  def connect(self):
    if len(self.chosen_points) != 2:
      print('Please only selected 2 points.')
      return

    self.chosen_points = sorted(self.chosen_points, key=lambda x:x[0])
    print(self.chosen_points)
    df = self.trajectories
    df[(df['frame'] < self.chosen_points[1][0]) & (df['particle'] == self.chosen_points[1][1])] = np.nan
    df[(df['frame'] > self.chosen_points[0][0]) & (df['particle'] == self.chosen_points[0][1])] = np.nan
    df.loc[(df['frame'] >= self.chosen_points[1][0])& (df['particle'] == self.chosen_points[1][1]), 'particle'] = self.chosen_points[0][1]
    df[(df['particle'] == self.chosen_points[1][1])] = np.nan
    print('Connected')
    self.chosen_points = []
    self.create_id = 9999

  def create(self):
    if self.mode != 'create':
      self.mode = 'create'
      self.create_id = self.trajectories['particle'].max() + 1
      if np.isnan(self.create_id):
        self.create_id = 0
    else:
      self.mode = 'select'
    print(f'Current mode {self.mode}')

  def delete(self):
    print(f'delete points {self.chosen_points}')
    for pt in self.chosen_points:
      df = self.trajectories
      df[(df['frame'] == pt[0]) & (
          df['particle'] == pt[1])] = np.nan

  def delete_all(self):
    print(f'delete all points with index {self.chosen_points}')
    for pt in self.chosen_points:
      df = self.trajectories
      df[df['particle'] == pt[1]] = np.nan

  def cancell_selected_points(self):
    print('Cancell selected points')
    self.chosen_points = []

  def select_point(self,event):
    ix, iy = event.xdata, event.ydata
    # print('x = %d, y = %d' % (ix, iy))
    if self.mode == 'select':
      df = self.trajectories
      df = df[df['frame'] == self.object_cursor]
      unstacked = df.set_index(['frame', 'particle'])[['x', 'y']].unstack()
      coords = unstacked.fillna(method='backfill').stack().loc[self.object_cursor]
      nearest_point_id = None
      distance_nearest = np.inf
      for particle_id, coord in coords.iterrows():
        distance = np.linalg.norm(np.array(coord) - np.array([ix, iy]))
        if distance < distance_nearest:
          distance_nearest = distance
          nearest_point_id = particle_id
      print(f'selected point{nearest_point_id} at frame{self.object_cursor}')
      self.chosen_points.append([self.object_cursor, nearest_point_id])
    elif self.mode == 'create':
      df = self.trajectories
      if len(df[(df['frame'] == self.object_cursor) & (df['particle'] == self.create_id)]) > 0:
        print(f'pt {self.create_id} has already been created at frame:{self.object_cursor}')
        return
      df.loc[len(df)] = np.nan
      df.loc[len(df)-1]['y'] = iy
      df.loc[len(df)-1]['x'] = ix
      df.loc[len(df)-1]['frame'] = self.object_cursor
      df.loc[len(df)-1]['particle'] = self.create_id
      print(f"create pt {self.create_id} at {ix, iy} at frame:{self.object_cursor}")

  def show_locations(self, show_traj=True, **locate_configs):
    # TODO: for some reason, locate config can be modified here
    # self.locate_configs has the highest priority

    locate_configs.update(self.locate_configs)

    # Calculate location if not exist
    if self.locations is None:
      self.locate(plot_progress=True, **locate_configs)
    configs = self.effective_locate_config
    df = self.locations
    # Link location if necessary
    if show_traj:
      if self.trajectories is None:
        self.link()
        self.trajectories = self.trajectories[0:0]
      configs = self.effective_link_config
      df = self.trajectories

    # Display different particle with different colors
    df = df[df['frame'] == self.object_cursor]
    # tp.annotate(df, self.raw_frames[self.object_cursor],
    #             color=np.vstack((np.linspace((0,0.99,0),(0,1,0),num=300))),
    #                             # np.linspace((0.5,0,0),(1,0,0),num=100),
    #                             # np.linspace((0,0,0.5),(0,0,1),num=100),)),
    #             split_category="particle",
    #             split_thresh=np.arange(299),ax=self.axes)
    self.imshow(self.raw_frames[self.object_cursor], self.axes)
    unstacked = df.set_index(['frame', 'particle'])[['x', 'y']].unstack()
    if len(unstacked) == 0: return self.trajectories
    coords = unstacked.fillna(method='backfill').stack().loc[self.object_cursor]
    for particle_id, coord in coords.iterrows():
      self.axes.text(*coord.tolist(), s="%d" % particle_id,color='red',
              horizontalalignment='center',
              verticalalignment='center')
    # Set title
    title = None
    if self.show_titles:
      title = ', '.join(['{} = {}'.format(k, v) for k, v in configs.items()])
      title += ' (#{})'.format(df.size)
    self.set_im_axes(title=title)

    return self.trajectories



if __name__ == '__main__':
  # data_dir = r'./data/mar2021'

  index =  2
  diameter = 11
  minmass = 1
  search_range= 10
  memory = 1
  invert=False
  #
  # diameter = 7
  # minmass = 3.5
  # search_range= 15
  # memory = 5
  # invert = True

  # Read the tif stack
  #tk = TrackerAlpha.read_by_index(data_dir, index, show_info=True)
  # file_name = 'adam'
  # file_name = 'unsyn_cos_2'
  file_name = 'T5P5_1'
  load = True
  save = False
  tk = TrackerAlpha.read(f'./data/{file_name}.tif', show_info=True)
  # tk.n_frames = 10
  if load:
    tk.trajectories = tk.locations = pandas
    # with open(f"./pkl/{file_name}.pkl", 'rb') as file:
    #   import pickle
    #   df = pickle.load(file)
    df = pandas.read_pickle(f"./pkl/{file_name}.pkl")
    # df[(df['particle'] > 250) & (df['particle'] < 450)] = df[0:0]
    # df[df['frame'] == 144] = df[df['frame'] == 144][0:0]
    tk.trajectories = tk.locations = df
  tk.config_locate(diameter=diameter,minmass=minmass, invert=invert)
  tk.config_link(search_range=search_range, memory=memory)
  tk.add_plotter(tk.imshow)
  tk.add_plotter(tk.show_locations)

  tk.show_locations()

  # print(df)
  df = tk.trajectories
  print(df)

  #save dataframe to pkl
  tk.show()
  if save:
    print('save to pickle')
    df.to_pickle(f"./pkl/{file_name}.pkl")
  #fig, ax = plt.subplots()

  #df_0 = df_0[df_0['frame'] == 0]
  #print(df_0)

  #ax.hist(df_0['raw_mass'], bins=20)
  # Optionally, label the axes.
  #ax.set(xlabel='raw_mass', ylabel='count');

  #tk.add_plotter(tk.pl)
