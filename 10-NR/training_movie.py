import json

from roma import console

import time
from tframe import console
console.suppress_logging()
from tframe import tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from guess_z import predict_z
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

def training(points, sess, step = 9999999):
  loss = loss_function(points)

  # Construct an optimizer
  optimizer = tf.train.GradientDescentOptimizer(5e-9)
  train_step = optimizer.minimize(loss)


  for t in range(step):
    sess.run(train_step)
    val_loss = sess.run(loss)

    # if val_loss < 30:
    # if t % 10000 == 0 and np.abs(pre_value_loss - val_loss) < 0.01:
    '''if val_loss < 500:
      print(f'Training compeleted at step {t} with loss:{val_loss}')
      with open('./z.json', 'w') as f:
        json.dump(sess.run(z).tolist(), f)
      with open('./x.json', 'w') as f:
        json.dump(sess.run(x).tolist(), f)
      with open('./y.json', 'w') as f:
        json.dump(sess.run(y).tolist(), f)
      break'''
    if t % 10000 == 0:
      print('loss = {}'.format(val_loss))
      pre_value_loss = val_loss

class DaVinci3D(DaVinci):
  def __init__(self, points, sess,  size: int = 10, **kwargs):
    # Call parent's constructor
    super(DaVinci3D, self).__init__('3d', height=size, width=size)
    self.points = points
    self.sess = sess
    self.predcit = self.sess.run(tf.transpose(self.points, perm=[0, 2, 1]))
    self.ground_truth = np.transpose(np.array(self.read('ground_true')), axes=(1, 0 ,2))
    self.num_anchors = len(self.ground_truth[0][0])
    self.objects = np.concatenate([self.predcit, self.ground_truth], axis=2)

  def draw_3D(self, x, ax3d: Axes3D):
    ax3d.scatter(*(x[:, 0:self.num_anchors]), 'ro', s=5)
    ax3d.scatter(*(x[:, self.num_anchors:]), 'bo', s=5)
    ax3d.set_xlim(-1.2, 1.2)
    ax3d.set_ylim(-1.2, 1.2)
    ax3d.set_zlim(-1.2, 1.2)

  def visualize_training_steps(self, steps=500000):
    for i in range(steps):
      console.print_progress(i, steps)
      self.update()
      self.refresh()

    console.show_status('Training completed.')
  go = visualize_training_steps

  def update(self):
    # Busy computing ...
    training(self.points, self.sess, 10000)
    self.predcit = self.sess.run(tf.transpose(self.points, perm=[0, 2, 1]))
    self.ground_truth = np.transpose(np.array(self.read('ground_true')), axes=(1, 0 ,2))
    self.objects = np.concatenate([self.predcit, self.ground_truth], axis=2)

  def read(self,path):
    x = json.load(open(f'./{path}/x.json'))
    y = json.load(open(f'./{path}/y.json'))
    z = json.load(open(f'./{path}/z.json'))
    return x,y, z

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

