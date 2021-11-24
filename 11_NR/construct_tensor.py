from frame import Frames, Frame
from tframe import tf


def frames_to_tensors(frames:list):

  points = Frames(frames).points

  tensor = 0
  pre_exist = 0
  # for each frame, we check the x, y of paricles and add it to the tensor
  for i in range(len(points)):
    current_tensor = 0
    frame = frames[i]
    assert isinstance(frame, Frame)
    for p in range(len(points[0])):
      if frame.missing[p] == 0 or True:
        # print("good, add trainbale Z")
        a = tf.Variable(initial_value=[[[points[i][p][0], points[i][p][1]]]],
                        trainable=False, dtype=tf.float64)
        b = tf.Variable(initial_value=[[[points[i][p][2]]]], trainable=True,
                        dtype=tf.float64)
        c = tf.concat([a, b], axis=2)
        if type(current_tensor) == int:
          current_tensor = c
        else:
          current_tensor = tf.concat([current_tensor,c], axis=1)

      else:

        print("shit! it disappears")
        if type(current_tensor) == int:
          current_tensor = tf.Variable(initial_value=[[[points[i][p][0],points[i][p][1],points[i][p][2]]]],
                                       trainable=True, dtype=tf.float64)
        else:
          current_tensor = tf.concat(
            [current_tensor, tf.Variable(initial_value=[[[points[i][p][0],points[i][p][1],points[i][p][2]]]],
                                         trainable=True, dtype=tf.float64)],
            axis=1)

    if type(tensor) == int:
      tensor = current_tensor
    else:
      tensor = tf.concat([tensor, current_tensor], axis=0)

  return tensor
