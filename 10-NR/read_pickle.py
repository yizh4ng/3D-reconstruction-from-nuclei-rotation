import json

import pandas as pd
import numpy as n

if __name__ == '__main__':
  data = pd.read_pickle('./data.pkl')
  data = data.filter(items = ['x', 'y', 'particle', 'frame'])
  data = data[data['particle'] == 0]
  print(data)