import json
import pickle

import tensorflow as tf
from tframe import console
console.suppress_logging()
from tframe import tf
from guess_z import predict_z
from tensor_padding import pad_df_to_tensors

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# A is an T * N * 3 size tenor out put the sum of distances of each pairs
# output T * N * N
def distance(A: tf.Tensor):
  # A.shape = (T, N, 3)
  A1 = tf.expand_dims(A, 2)
  A2 = tf.expand_dims(A, 1)
  dis = tf.norm(A1 - A2 ,axis=-1)
  dis = tf.reduce_sum(tf.abs(A1 - A2), axis=-1)
  return dis
  # dis.shape = (T, N, N)

# points: T * N * 3 size tensor
def loss_function(points):
  # distances: T * N * N
  D = distance(points)
  # pair_dif = T * T * N * N
  pair_dif = tf.abs((tf.expand_dims(D, 1) - tf.expand_dims(D,0)))
  # pair_dif = tf.norm((tf.expand_dims(D, 1) - tf.expand_dims(D,0)))
  output = tf.reduce_sum(pair_dif)
  return output

def training(points, sess, step = 9999999):
  try:
    tf.debugging.check_numerics(points,message='')
  except Exception as e:
    assert "Tensor had NaN values" in e.message
  loss = loss_function(points)

  # Construct an optimizer
  optimizer = tf.train.GradientDescentOptimizer(1e-8)
  train_step = optimizer.minimize(loss)

  # Initialize w in GPU
  sess.run(tf.global_variables_initializer())
  print('[0] loss = {}'.format(sess.run(loss)))

  for t in range(step):
    sess.run(train_step)
    val_loss = sess.run(loss)

    # if val_loss < 30:
    # if t % 10000 == 0 and np.abs(pre_value_loss - val_loss) < 0.01:
    if val_loss < 81:
      print(f'Training compeleted at step {t} with loss:{val_loss}')
      with open('./train_result.json', 'w') as f:
        json.dump(sess.run(points).tolist(), f)
      break
    if t % 100 == 0:
      print('[{}] loss = {}'.format(t + 1, val_loss))

if __name__ == '__main__':
  f = open('data.pkl', 'rb')
  df = pickle.load(f)
  df = df[df['frame'] < 10]
  df = df[df['z'] > 0]

  df.drop('z', axis=1, inplace=True)
  df['z'] = 0
  predict_z(df)
  # print(get_particle_list(df)[1])
  tensor = pad_df_to_tensors(df)
  sess = tf.Session()
  training(tensor, sess)
