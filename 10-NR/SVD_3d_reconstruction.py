import json
import math_utili.solve
import math_utili.SVD
import numpy as np
import numpy.linalg as LA
from visualize_xyz import visualize_xyz
from lambo.gui.vinci.vinci import DaVinci

if __name__ == '__main__':
  x = json.load(open('./ground_truth/x.json'))
  y = json.load(open('./ground_truth/y.json'))
  M = math_utili.build_matrix.build_matrix(x,y)
  P, E, Q = math_utili.SVD.SV_D(M)
  P, E, Q = math_utili.SVD.select_3_most_SV(P, E, Q)
  sol = math_utili.solve.solve_x_y(x, y, P, E)
  print(sol)
  Q = np.matmul(np.sqrt(E), Q)
  s = np.expand_dims(np.matmul(LA.inv(sol), Q), axis=0)

  vis = visualize_xyz()
  vis.objects = np.array(s)
  vis.add_plotter(vis.draw_3d)
  vis.show()