import tensorflow as tf

if __name__ == '__main__':
  x = tf.constant([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=tf.float32) # 4 * 2
  y = tf.constant([[0, 1,], [2, 3]],dtype=tf.float32) # 2 * 2

  x_ =  tf.expand_dims(x, 0) # 1 * 4 * 2
  y_ = tf.expand_dims(y, 1) # 2 * 1 * 2
  z = tf.add(x_, y_)
  print(z)
  '''z = tf.reshape(tf.add(x_, y_), [-1, 2])
  print(x_, y_)
  z = tf.add(x, y)
  print(z)
  #print(x)'''