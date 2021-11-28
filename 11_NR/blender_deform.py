import bpy
import bmesh
from pathlib import Path
import pickle
import numpy as np

# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete(use_global=False)

path = "C:/Users/Administrator/Desktop/lambai/11_NR/results/adam/cell_op.pkl"
with open(path, 'rb') as f:
  dict = pickle.load(f)
dict['x'] = np.concatenate((np.array(dict['x']),
                            np.expand_dims(2 * np.array(dict['center'])[:, 0],
                                           axis=-1) - np.array(dict['x'])),
                           axis=1)
dict['y'] = np.concatenate((np.array(dict['y']),
                            np.expand_dims(2 * np.array(dict['center'])[:, 1],
                                           axis=-1) - np.array(dict['y'])),
                           axis=1)
dict['z'] = np.concatenate((np.array(dict['z']), - np.array(dict['z'])),
                           axis=1)
alpha = 1
for index in range(len(dict['x'])):

  if index < 70 or index > 140: continue
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
  bmesh.ops.create_icosphere(bm, subdivisions=3, diameter=1 * dict['radius'])
  bm.to_mesh(mesh)
  bm.free()
  mymat = bpy.data.materials.new("Plane_mat")
  mymat.diffuse_color = (0.6, 1, 1, 1)
  mesh.materials.append(mymat)
  # bpy.ops.object.modifier_add(type='SUBSURF')
  bpy.ops.object.shade_smooth()

  mesh = bpy.data.meshes[f'sphere{index}']
  basic_sphere.select_set(False)
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
    nearest_point = np.array(nearest_point)[np.argsort(length)[:5]]
    nearest.append(nearest_point)

  nucleis = []
  for i, verts in enumerate(nearest):
    base = verts[0].co
    offset = [(x[i] - center[0]) - base.x,
              (y[i] - center[1]) - base.y,
              (z[i] - 0) - base.z]
    for j, vert in enumerate(verts):
      #      print(offset)

      l = np.linalg.norm(np.array([base.x - vert.co.x,
                                   base.y - vert.co.y,
                                   base.z - vert.co.z]))
      vert.co.x += alpha * (1 / (1 + l)) * offset[0]
      vert.co.y += alpha * (1 / (1 + l)) * offset[1]
      vert.co.z += alpha * (1 / (1 + l)) * offset[2]
  for i, verts in enumerate(nearest):
    if i == len(nearest) / 2: break
    mesh_ = bpy.data.meshes.new(f'sphere{index}-{i}')
    nuclei = bpy.data.objects.new(f"Sphere{index}-{i}", mesh_)
    bpy.context.collection.objects.link(nuclei)

    # Select the newly created object
    bpy.context.view_layer.objects.active = nuclei
    bm = bmesh.new()
    # bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=32, diameter= 0.95 * dict['radius'])
    bmesh.ops.create_icosphere(bm, subdivisions=1,
                               diameter=0.1 * dict['radius'])
    bm.to_mesh(mesh_)
    bm.free()
    mymat = bpy.data.materials.new("Plane_mat")
    mymat.diffuse_color = (1, 1, 0.6, 1)
    mesh_.materials.append(mymat)
    nuclei.select_set(True)
    bpy.ops.transform.translate(
      value=np.linalg.norm([verts[0].co.x, verts[0].co.y, verts[0].co.z])
            * np.array(
        [x[i] - center[0], y[i] - center[1], z[i]]) / np.linalg.norm(
        [x[i] - center[0], y[i] - center[1], z[i]]))
    nuclei.select_set(False)

    nucleis.append(nuclei)

  for f in range(len(dict['x'])):
    if f == index:
      # key as hidden on the next frame
      basic_sphere.hide_viewport = False
      basic_sphere.hide_render = False
      for nuclei in nucleis:
        nuclei.hide_viewport = False
        nuclei.hide_render = False
        nuclei.keyframe_insert('hide_viewport', frame=f)
        nuclei.keyframe_insert('hide_render', frame=f)
      basic_sphere.keyframe_insert('hide_viewport', frame=f)
      basic_sphere.keyframe_insert('hide_render', frame=f)
    if f != index:
      basic_sphere.hide_viewport = True
      basic_sphere.hide_render = True
      for nuclei in nucleis:
        nuclei.hide_viewport = True
        nuclei.hide_render = True
        nuclei.keyframe_insert('hide_viewport', frame=f)
        nuclei.keyframe_insert('hide_render', frame=f)
      basic_sphere.keyframe_insert('hide_viewport', frame=f)
      basic_sphere.keyframe_insert('hide_render', frame=f)