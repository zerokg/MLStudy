import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

hello = tf.constant("Hello, Tensorflow!")

sess = tf.Session()

print(sess.run(hello))
