import pandas as pd

class Frames():
  def __init__(self, df: pd.DataFrame):
    pass
  def get_particle_list(self, df:pd.DataFrame):
    return list(set(list(map(int, df['frame'].values))))


