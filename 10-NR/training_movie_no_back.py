import json
import pickle

from roma import console
import pandas as pd
import time
from tframe import console
console.suppress_logging()
from tframe import tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from guess_z import predict_z
from lambo import DaVinci
from tensor_padding import pad_df_to_tensors
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
  # dif = tf.abs(D[0:-1] - D[1:])
  pair_dif = tf.reduce_sum(
    tf.abs((tf.expand_dims(D, 1) - tf.expand_dims(D,0))))
  frame_dif = tf.reduce_sum(tf.abs(points[0:-1] - points[1:]))
  #output = tf.reduce_sum(pair_dif)
  return pair_dif + 100 * frame_dif

def training(points, sess, step = 9999999):
  loss = loss_function(points)

  # Construct an optimizer
  optimizer = tf.train.GradientDescentOptimizer(5e-5)
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
    #self.ground_truth = np.transpose(np.array(self.read('data.pkl')), axes=(1, 0 ,2))
    #self.num_anchors = len(self.ground_truth[0][0])
    #self.objects = np.concatenate([self.predcit, self.ground_truth], axis=2)
    self.objects= self.predcit

  def draw_3D(self, x, ax3d: Axes3D):
    #ax3d.scatter(*(x[:, 0:self.num_anchors]), 'ro', s=5)
    #ax3d.scatter(*(x[:, self.num_anchors:]), 'bo', s=5)
    ax3d.scatter(*x, 'bo', s=5)
    #ax3d.set_xlim(-1.2, 1.2)
    #ax3d.set_ylim(-1.2, 1.2)
    #ax3d.set_zlim(-1.2, 1.2)

  def visualize_training_steps(self, steps=10):
    for i in range(steps):
      console.print_progress(i, steps)
      self.update()
      self.refresh()

    console.show_status('Training completed.')
  go = visualize_training_steps

  def update(self):
    # Busy computing ...
    training(self.points, self.sess, 1000)
    self.predcit = self.sess.run(tf.transpose(self.points, perm=[0, 2, 1]))
    #self.ground_truth = np.transpose(np.array(self.read('ground_true')), axes=(1, 0 ,2))
    #self.objects = np.concatenate([self.predcit, self.ground_truth], axis=2)
    self.objects= self.predcit

  def save_train_result(self):
    file = open('train_result.json','w')
    json.dump(self.sess.run(self.points).tolist(), file)
  save= save_train_result

  def read(self, path):
    f = open(path, 'rb')
    df: pd.DataFrame = pickle.load(f)
    x, y, z = [], [], []
    for f in range(int(df['frame'].max()) + 1):
      x.append(df[df['frame'] == f]['x'].tolist())
      y.append(df[df['frame'] == f]['y'].tolist())
      z.append(df[df['frame'] == f]['z'].tolist())
    return x, y, z

if __name__ == '__main__':
  f = open('data_real.pkl', 'rb')
  df = pickle.load(f)
  df = df[df['frame']<10]
  #df = df[df['z'] > 0]
  #df.drop('z', axis=1, inplace=True)
  df['z'] = 0
  predict_z(df)
  # print(get_particle_list(df)[1])
  points = pad_df_to_tensors(df, track_disappear_achors=True)

  sess = tf.Session()

  sess.run(tf.global_variables_initializer())

  da = DaVinci3D(points, sess)

  # One points in each frame]
  da.add_plotter(da.draw_3D)

  da.keep_3D_view_angle = True
  da.show()

