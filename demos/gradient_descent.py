import tensorflow as tf

def fu(x1, x2):
	return x1 ** 2.0 - x1 * 3  + x2 ** 2

def fu_minimzie():
	return x1 ** 2.0 - x1 * 3  + x2 ** 2

def reset():
	x1 = tf.Variable(10.0)
	x2 = tf.Variable(10.0)
	return x1, x2

x1, x2 = reset()

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
print(fu_minimzie(), [x1, x2])
for i in range(50):
	print ('y = {:.1f}, x1 = {:.1f}, x2 = {:.1f}'.format(fu(x1, x2).numpy(), x1.numpy(), x2.numpy()))
	opt.minimize(fu_minimzie, var_list=[x1, x2])
