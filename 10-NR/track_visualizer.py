from pandas import DataFrame
from tracker_alpha import TrackerAlpha
import numpy as np
from matplotlib import pyplot as plt
def int_to_RGB(index:int):
  colors = np.vstack((np.linspace((0, 0.5, 0), (0, 1, 0), num=100),
                     np.linspace((0.5, 0, 0), (1, 0, 0), num=100),
                     np.linspace((0, 0, 0.5), (0, 0, 1), num=100),)),
  return colors[index]

def get_anchor_num(df):
    return df["particle"].max()

def draw_track(df:DataFrame):
    fig, ax = plt.subplots()
    for i in range(get_anchor_num(df)):
        df[df['particle'] == i].plot(x='x', y = 'y',ax = ax)

def draw_track_v1(df:DataFrame):
    fig, ax = plt.subplots()
    for i in range(get_anchor_num(df)):
        plt.plot(df[df['particle'] == i]['x'],
                 df[df['particle'] == i]['y'],
                 linewidth=0.5)



if __name__ == '__main__':
  data_dir = r'./data/mar2021'

  index = 2
  diameter = 7
  minmass = 0.7

  # Read the `index`-th tif stack
  tk = TrackerAlpha.read_by_index(data_dir, index, show_info=True)

  tk.config_locate(diameter=diameter,minmass=minmass)
  tk.config_link(search_range=30, memory=2)
  df = tk.show_locations()
  draw_track_v1(df)
  plt.show()
