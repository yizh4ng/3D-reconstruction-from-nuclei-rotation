import json

import tensorflow as tf
from tframe import console
console.suppress_logging()
from tframe import tf
from data_cleasing import read_x_y_from_pkl
import numpy as np
import os
from guess_z import predict_z
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# A is an T * N * 3 size tenor out put the sum of distances of each pairs
# output T * N * N
def distance(A: tf.Tensor):
  # A.shape = (T, N, 3)
  A1 = tf.expand_dims(A, 2)
  A2 = tf.expand_dims(A, 1)
  dis = tf.norm(A1 - A2 ,axis=-1)
  return dis
  # dis.shape = (T, N, N)

# points: T * N * 3 size tensor
def loss_function(points):
  # distances: T * N * N
  D = distance(points)
  # pair_dif = T * T * N * N
  pair_dif = tf.abs((tf.expand_dims(D, 1) - tf.expand_dims(D,0)))
  output = tf.reduce_sum(pair_dif)
  return output

def training(points, sess, step = 9999999):
  loss = loss_function(points)

  # Construct an optimizer
  optimizer = tf.train.GradientDescentOptimizer(1e-6)
  train_step = optimizer.minimize(loss)

  # Initialize w in GPU
  sess.run(tf.global_variables_initializer())
  print('[0] loss = {}'.format(sess.run(loss)))
  print(sess.run(z))

  for t in range(step):
    sess.run(train_step)
    val_loss = sess.run(loss)

    # if val_loss < 30:
    # if t % 10000 == 0 and np.abs(pre_value_loss - val_loss) < 0.01:
    if val_loss < 500:
      print(f'Training compeleted at step {t} with loss:{val_loss}')
      print('z = {}'.format(sess.run(z)))
      with open('./z.json', 'w') as f:
        json.dump(sess.run(z).tolist(), f)
      with open('./x.json', 'w') as f:
        json.dump(sess.run(x).tolist(), f)
      with open('./y.json', 'w') as f:
        json.dump(sess.run(y).tolist(), f)
      break
    if t % 10000 == 0:
      print('[{}] loss = {}'.format(t + 1, val_loss))
      print('z = {}'.format(sess.run(z)))
      pre_value_loss = val_loss

if __name__ == '__main__':
  max_particle = 100
  max_frame = 100
  #x, y = read_x_y_from_pkl('data_3.pkl', max_particle=20, max_frame=3
  x = json.load(open('./ground_true/x.json'))
  y = json.load(open('./ground_true/y.json'))
  x = np.array(x)[0:max_frame, 0:max_particle].tolist()
  y = np.array(y)[0:max_frame, 0:max_particle].tolist()
  num_frames = len(x)
  initial_values = np.array([predict_z(x, y), ] * len(x))
  print(initial_values)

  '''# fix the first anchor in the first frame to avoid translation
  z1 = tf.Variable(initial_value=[[initial_values.tolist()[0][0]]],trainable=False, dtype=tf.float64)
  z2 = tf.Variable(initial_value=[initial_values.tolist()[0][1:]], trainable=True, dtype=tf.float64)
  z1_ = tf.concat([z1, z2],axis=-1)
  z2_ = tf.Variable(initial_value=initial_values.tolist()[1:],trainable=True,dtype=tf.float64)
  z = tf.concat([z1_, z2_], axis=0)'''
  # fix the fix anchors in all frames
  z1 = tf.Variable(initial_value=np.reshape(np.array(initial_values.tolist())[:,0], (len(x),1)).tolist(), trainable=False, dtype=tf.float64)
  z2 = tf.Variable(initial_value=np.array(initial_values.tolist())[:, 1:].tolist(), trainable=True, dtype=tf.float64)
  z = tf.concat([z1,z2], axis=1)
  #z = tf.Variable(initial_value=initial_values.tolist(), trainable=True)
  x = tf.constant(x, dtype=tf.float64)
  y = tf.constant(y, dtype=tf.float64)

  points = tf.transpose([x, y, z], perm=[1, 2, 0])
  sess = tf.Session()
  training(points, sess)