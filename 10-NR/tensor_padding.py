import pickle
import tensorflow as tf
import pandas as pd
from guess_z import predict_z
# To convert the df to T * N * 3 size Tensor

def get_particle_list(df: pd.DataFrame):
  particle_list = []
  for index, row in df.iterrows():
    if int(row['particle']) not in particle_list:
      particle_list.append(int(row['particle']))

  particle_list.sort()
  return particle_list

def pad_df_to_tensors(df: pd.DataFrame, track_disappear_achors = True):
  particle_list = get_particle_list(df)
  num_anchors = len(particle_list)
  current_particle = 0
  current_frame = 0
  current_tensor = 0
  tensor = 0

  for index, row in df.iterrows():
    #print(row)
    if current_particle >= num_anchors:
      #print("oh, new frame!")
      current_frame += 1
      current_particle = 0
      if type(tensor) == int:
        tensor = current_tensor
      else:
        tensor = tf.concat([tensor, current_tensor], axis=0)
      current_tensor = 0

    if int(row['particle']) != particle_list[current_particle]:
      while row['particle'] != particle_list[current_particle]:
        #print("shit! Particle disappear, add trainable x, y, z")
        if type(current_tensor) == int:
          current_tensor = tf.Variable(initial_value=[[[row['x'],row['y'],
                                      row['z']]]], trainable=True)
        else:
          current_tensor = tf.concat([current_tensor,tf.Variable(initial_value=[[[row['x'],row['y'],
                                      row['z']]]], trainable=True)], axis=1)
        current_particle += 1
      if int(row['particle']) == particle_list[current_particle] and int(
        row['frame']) == current_frame:
        #print("good, add a trainable z")

        a = tf.Variable(initial_value=[[[row['x'], row['y']]]], trainable=False)
        b = tf.Variable(initial_value=[[[row['z']]]], trainable=True)
        c = tf.concat([a, b], axis=2)

        if type(current_tensor) == int:
          current_tensor = c
        else:
          current_tensor = tf.concat([current_tensor, c], axis=1)

        current_particle += 1

    elif int(row['particle']) == particle_list[current_particle] and int(row['frame']) == current_frame:
      #print("good, add a trainable z")

      a = tf.Variable(initial_value=[[[row['x'], row['y']]]], trainable=False)
      b = tf.Variable(initial_value=[[[row['z']]]], trainable=True)
      c = tf.concat([a, b], axis=2)

      if type(current_tensor) == int:
        current_tensor = c
      else:
        current_tensor = tf.concat([current_tensor, c], axis=1)

      current_particle += 1

    else:
      print("WTF?", row['particle'], current_particle)

  return tensor


if __name__ == '__main__':
  f = open('data.pkl', 'rb')
  df = pickle.load(f)
  df = df[df['z'] > 0]
  df.drop('z', axis=1, inplace=True)
  df['z'] = 0
  predict_z(df)
  # print(get_particle_list(df)[1])
  tensor = pad_df_to_tensors(df)
  print(tensor)
  grad_check = tf.debugging.check_numerics(tensor,
                                 'check_numerics caught bad gradients')
