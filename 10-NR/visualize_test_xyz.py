from visualize_xyz import visualize_xyz
if __name__ == '__main__':
  file = 'test_7'
  vis_2 = visualize_xyz()
  vis_2.keep_3D_view_angle = True
  x, y, z = vis_2.read(file)
  vis_2.vis_animation(x, y, z)
  vis_2.show()