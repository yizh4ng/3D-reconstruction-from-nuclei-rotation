from lambo.analyzer.tracker import Tracker

import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt
class TrackerAlpha(Tracker):

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
      if self.trajectories is None: self.link()
      configs = self.effective_link_config
      df = self.trajectories
    # Display different particle with different colors
    df = df[df['frame'] == self.object_cursor]
    tp.annotate(df, self.raw_frames[self.object_cursor],
                color=np.vstack((np.linspace((0,0.5,0),(0,1,0),num=100),
                                np.linspace((0.5,0,0),(1,0,0),num=100),
                                np.linspace((0,0,0.5),(0,0,1),num=100),)),
                split_category="particle",
                split_thresh=np.arange(299),ax=self.axes)
    # Set title
    title = None
    if self.show_titles:
      title = ', '.join(['{} = {}'.format(k, v) for k, v in configs.items()])
      title += ' (#{})'.format(df.size)
    self.set_im_axes(title=title)

    return self.trajectories
  pass



if __name__ == '__main__':
  data_dir = r'./data/mar2021'

  index =  2
  diameter =11
  minmass = 0.5

  # Read the tif stack
  #tk = TrackerAlpha.read_by_index(data_dir, index, show_info=True)
  file_name = 'data_8'
  save = False
  tk = TrackerAlpha.read(f'./data/{file_name}.tif', show_info=True)
  # tk.n_frames = 30
  tk.config_locate(diameter=diameter,minmass=minmass)
  tk.config_link(search_range=10, memory=15)
  tk.add_plotter(tk.imshow)
  tk.add_plotter(tk.show_locations)

  df = tk.show_locations()
  print(df)
  #save dataframe to pkl
  if save:
    df.to_pickle(f"./pkl/{file_name}.pkl")
  #fig, ax = plt.subplots()

  #df_0 = df_0[df_0['frame'] == 0]
  #print(df_0)

  #ax.hist(df_0['raw_mass'], bins=20)
  # Optionally, label the axes.
  #ax.set(xlabel='raw_mass', ylabel='count');

  #tk.add_plotter(tk.pl)
  tk.show()
