import json

from roma import console

import time
from tframe import console
console.suppress_logging()
from tframe import tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from frame import Frames
from lambo import DaVinci

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

def training(points, sess, step = 2):

  loss = loss_function(points)

  # Construct an optimizer
  optimizer = tf.train.GradientDescentOptimizer(5e-9)
  train_step = optimizer.minimize(loss)


  for t in range(step):
    sess.run(train_step)
    val_loss = sess.run(loss)

    # if val_loss < 30:
    # if t % 10000 == 0 and np.abs(pre_value_loss - val_loss) < 0.01:
    if val_loss < 500:
      # print(f'Training compeleted at step {t} with loss:{val_loss}')
      # with open('./z.json', 'w') as f:
      #   json.dump(sess.run(z).tolist(), f)
      # with open('./x.json', 'w') as f:
      #   json.dump(sess.run(x).tolist(), f)
      # with open('./y.json', 'w') as f:
      #   json.dump(sess.run(y).tolist(), f)
      break
    if t % 10000 == 0:
      print('loss = {}'.format(val_loss))
      pre_value_loss = val_loss

class Trainer():
  def __init__(self):
    pass

  @classmethod
  def train(cls, frames:list):
    frames_ = Frames(frames)
    points = frames_.points
    x = tf.constant(points[:, :, 0])
    y = tf.constant(points[:, :, 1])
    print(points[:, :, 2])
    z = tf.Variable(initial_value=points[:, :, 2], trainable=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    points = tf.concat([x, y, z], axis=-1)
    training(points, sess)
    predcit = sess.run(points)
    frames_.set_points(predcit)
    return frames_.frames

if __name__ == '__main__':
  '''da = DaVinci3D('3D Movie')

  # One points in each frame
  da.objects = [[0, 0, 1], [0, 1, 0]]
  da.add_plotter(da.draw_3D)

  da.keep_3D_view_angle = True
  da.show()'''
  max_particle = 100
  max_frame = 100
  x = json.load(open('./ground_true/x.json'))
  y = json.load(open('./ground_true/y.json'))
  x = np.array(x)[0:max_frame, 0:max_particle].tolist()
  y = np.array(y)[0:max_frame, 0:max_particle].tolist()
  initial_values = np.array([predict_z(x, y), ] * len(x))

  # fix the first anchor in the first frame to avoid translation
  z1 = tf.Variable(initial_value=[[initial_values.tolist()[0][0]]],trainable=False, dtype=tf.float64)
  z2 = tf.Variable(initial_value=[initial_values.tolist()[0][1:]], trainable=True, dtype=tf.float64)
  z1_ = tf.concat([z1, z2],axis=-1)
  z2_ = tf.Variable(initial_value=initial_values.tolist()[1:],trainable=True,dtype=tf.float64)
  z = tf.concat([z1_, z2_], axis=0)
  x = tf.constant(x, dtype=tf.float64)
  y = tf.constant(y, dtype=tf.float64)
  points = tf.transpose([x, y, z], perm=[1, 2, 0])

  sess = tf.Session()

  sess.run(tf.global_variables_initializer())

  da = DaVinci3D(points, sess)

  # One points in each frame]
  da.add_plotter(da.draw_3D)

  da.keep_3D_view_angle = True
  da.show()

