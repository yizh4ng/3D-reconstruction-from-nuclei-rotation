import json

from visualize_xyz import visualize_xyz
if __name__ == '__main__':
  vis_2 = visualize_xyz()
  f = open('train_result.json', 'rb')
  vis_2.read_x_y_z_json(json.load(f))
  vis_2.add_plotter(vis_2.draw_3d)
  vis_2.show()