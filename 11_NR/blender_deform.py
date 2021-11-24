import bpy
import bmesh
from pathlib import Path
import pickle
import numpy as np

# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete(use_global=False)

path = "C:/Users/Administrator/Desktop/lambai/11_NR/results/adam/cell_176.pkl"
with open(path, 'rb') as f:
  dict = pickle.load(f)
alpha = 1
for index in range(len(dict['x'])):

  #  if index > 3: break
  # Create an empty mesh and the object.
  mesh = bpy.data.meshes.new(f'sphere{index}')
  basic_sphere = bpy.data.objects.new(f"Sphere{index}", mesh)

  # Add the object into the scene.
  bpy.context.collection.objects.link(basic_sphere)

  # Select the newly created object
  bpy.context.view_layer.objects.active = basic_sphere
  basic_sphere.select_set(True)

  # Construct the bmesh sphere and assign it to the blender mesh.
  bm = bmesh.new()
  # bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=32, diameter= 0.95 * dict['radius'])
  bmesh.ops.create_icosphere(bm, subdivisions=2, diameter=1 * dict['radius'])
  bm.to_mesh(mesh)
  bm.free()

  bpy.ops.object.modifier_add(type='SUBSURF')
  bpy.ops.object.shade_smooth()

  mesh = bpy.data.meshes[f'sphere{index}']

  # bpy.data.meshes.remove(mesh)
  x = dict['x'][index]
  y = dict['y'][index]
  z = dict['z'][index]
  center = dict['center'][index]
  # print(center)
  nearest = []
  for i in range(len(x)):
    #  print(i)
    nearest_point = []
    length = []
    for vert in mesh.vertices:
      l = np.linalg.norm(
        np.array([x[i] - center[0] - vert.co.x, y[i] - center[1] - vert.co.y,
                  z[i] - 0 - vert.co.z]))
      length.append(l)
      nearest_point.append(vert)
    #    print(np.sort(length)[:10])
    nearest_point = np.array(nearest_point)[np.argsort(length)[:10]]
    nearest.append(nearest_point)

  for i, verts in enumerate(nearest):
    base = verts[0].co
    offset = [(x[i] - center[0]) - base.x,
              (y[i] - center[1]) - base.y,
              (z[i] - 0) - base.z]
    for vert in verts:
      #      print(offset)
      l = np.linalg.norm(np.array([base.x - vert.co.x,
                                   base.y - vert.co.y,
                                   base.z - vert.co.z]))
      vert.co.x += alpha * (1 / (1 + l)) * offset[0]
      vert.co.y += alpha * (1 / (1 + l)) * offset[1]
      vert.co.z += alpha * (1 / (1 + l)) * offset[2]

  for f in range(len(dict['x'])):
    if f == index:
      # key as hidden on the next frame
      basic_sphere.hide_viewport = False
      basic_sphere.hide_render = False
      basic_sphere.keyframe_insert('hide_viewport', frame=f)
      basic_sphere.keyframe_insert('hide_render', frame=f)
    if f != index:
      basic_sphere.hide_viewport = True
      basic_sphere.hide_render = True
      basic_sphere.keyframe_insert('hide_viewport', frame=f)
      basic_sphere.keyframe_insert('hide_render', frame=f)