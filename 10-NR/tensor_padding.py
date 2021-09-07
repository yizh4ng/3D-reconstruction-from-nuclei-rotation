import pickle
import tensorflow as tf
import pandas as pd
from guess_z import predict_z
from data_cleasing import remove_unlink
# To convert the df to T * N * 3 size Tensor

def get_particle_list(df: pd.DataFrame):
  particle_list = []
  for index, row in df.iterrows():
    if int(row['particle']) not in particle_list:
      particle_list.append(int(row['particle']))

  particle_list.sort()
  return particle_list

def pad_df_to_tensors(df: pd.DataFrame, track_disappear_achors = True):
  df = df.sort_values(by=['frame', 'particle'])
  if not track_disappear_achors:
    df = remove_unlink(df)

  particle_list = get_particle_list(df)


  tensor = 0
  pre_exist = 0
  for i in range(int(df['frame'].max()) + 1):
    frame = df[df['frame'] == i]
    print(f"new frame!{i}")
    current_tensor = 0

    for p in range(len(particle_list)):
      if particle_list[p] not in map(int, frame['particle'].tolist()):
        print("shit! it disappears")
        row = pre_exist
        assert row.shape[0] == 1
        row = row.iloc[0]
        a = tf.Variable(initial_value=[[[row['x'], row['y']]]],
                        trainable=False, dtype=tf.float64)
        b = tf.Variable(initial_value=[[[-row['z']]]], trainable=True, dtype=tf.float64)
        c = tf.concat([a, b], axis=2)
        if type(current_tensor) == int:
          current_tensor = c
        else:
          current_tensor = tf.concat([current_tensor,c], axis=1)

      else:
        print("good, add trainbale Z")
        row = frame[frame['particle'] == particle_list[p]]
        pre_exist = row
        assert row.shape[0] == 1
        row = row.iloc[0]
        if type(current_tensor) == int:
          current_tensor = tf.Variable(initial_value=[[[row['x'], row['y'],
                                                        row['z']]]],
                                       trainable=True, dtype=tf.float64)
        else:
          current_tensor = tf.concat(
            [current_tensor, tf.Variable(initial_value=[[[row['x'], row['y'],
                                                          row['z']]]],
                                         trainable=True, dtype=tf.float64)], axis=1)

    if type(tensor) == int:
      tensor = current_tensor
    else:
      tensor = tf.concat([tensor, current_tensor], axis=0)

  return tensor

if __name__ == '__main__':
  f = open('data.pkl', 'rb')
  df = pickle.load(f)
  df = df[df['z'] > 0]
  df.drop('z', axis=1, inplace=True)
  df['z'] = 0
  predict_z(df)
  # print(get_particle_list(df)[1])
  tensor = pad_df_to_tensors(df, track_disappear_achors=False)
  # print(tensor)
