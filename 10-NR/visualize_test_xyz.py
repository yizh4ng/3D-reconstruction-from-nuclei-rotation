from visualize_xyz import visualize_xyz
if __name__ == '__main__':
  vis_2 = visualize_xyz()
  f = open('train_result.json', 'rb')
  x, y, z = vis_2.read_json(f)
  vis_2.vis_animation(x, y, z)
  vis_2.show()