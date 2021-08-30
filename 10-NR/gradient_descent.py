import json

import tensorflow as tf
from tframe import console
console.suppress_logging()
from tframe import tf
from data_cleasing import read_x_y_from_pkl
import numpy as np
import os
from guess_z import predict_z
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from lambo.gui.vinci.vinci import DaVinci

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def distance( x, y, z, id_1, id_2, frame_number):
  x1, x2 = x[frame_number][id_1],x[frame_number][id_2]
  y1, y2 = y[frame_number][id_1], y[frame_number][id_2]
  z1, z2 = z[frame_number][id_1], z[frame_number][id_2]
  return tf.math.sqrt((x1 - x2) ** 2 +(y1 - y2) ** 2 + (z1 - z2) ** 2)

def loss_function(x, y, z):
  loss = 0
  num_anchors = len(x[0])
  num_frames = len(x)
  for i in range(num_anchors):
    for j in range(i+1, num_anchors):
      for t in range(0, num_frames - 1):
        loss += tf.math.abs(
          distance(x, y, z, i, j, t) - distance(x, y, z, i, j, t + 1))# + \
                #0.1 * tf.norm(z[t] - z[t + 1])
  return loss

def loss_function_2(x, y, z):
  loss = 0
  num_anchors = len(x[0])
  num_frames = len(x)
  for i in range(num_anchors):
    for j in range(i+1, num_anchors):
      for t in range(0, num_frames - 1):
        loss += tf.math.abs(
          distance(x, y, z, i, j, t) - distance(x, y, z, i, j, t + 1))
  return loss


if __name__ == '__main__':
  max_particle = 5
  max_frame = 3
  #x, y = read_x_y_from_pkl('data_3.pkl', max_particle=20, max_frame=3
  x = json.load(open('./ground_true/x.json'))
  y = json.load(open('./ground_true/y.json'))
  x = np.array(x)[0:max_frame, 0:max_particle].tolist()
  y = np.array(y)[0:max_frame, 0:max_particle].tolist()
  #initial_values = 200 * np.random.random(np.array(x).shape) + 200
  initial_values = np.array([predict_z(x, y), ] * len(x))
  print(initial_values)

  # fix the first anchor in the first frame to avoid translation
  z1 = tf.Variable(initial_value=[[initial_values.tolist()[0][0]]],trainable=False)
  z2 = tf.Variable(initial_value=[initial_values.tolist()[0][1:]], trainable=True)
  z1_ = tf.concat([z1, z2],axis=-1)
  z2_ = tf.Variable(initial_value=initial_values.tolist()[1:],trainable=True)
  z = tf.concat([z1_, z2_], axis=0)
  #z = tf.Variable(initial_value=initial_values.tolist(), trainable=True)

  # Initialize a session
  sess = tf.Session()
  loss = loss_function(x, y, z)
  loss_2 = loss_function_2(x, y, z)

  # Construct an optimizer
  optimizer = tf.train.GradientDescentOptimizer(0.0002)
  train_step = optimizer.minimize(loss)



  # Initialize w in GPU
  sess.run(tf.global_variables_initializer())
  print('[0] loss = {}'.format(sess.run(loss)))
  print(sess.run(z))

  pre_value_loss = 99999999
  for t in range(99999999):
    sess.run(train_step)
    val_loss = sess.run(loss_2)

    #if val_loss < 30:
    #if t % 10000 == 0 and np.abs(pre_value_loss - val_loss) < 0.01:
    if t % 10000 == 0 and val_loss < 0.01:
      print(val_loss)
      print('z = {}'.format(sess.run(z)))
      from datetime import datetime

      now = datetime.now()
      current_time = now.strftime("%H:%M:%S")
      with open('./z.json', 'w') as f:
        json.dump(sess.run(z).tolist(), f)
      with open('./x.json', 'w') as f:
        json.dump(x, f)
      with open('./y.json', 'w') as f:
        json.dump(y, f)
      break
    if t % 10000 == 0:
      print('[{}] loss = {}'.format(t + 1, val_loss))
      print('z = {}'.format(sess.run(z)))
      pre_value_loss = val_loss