import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

a = tf.Variable(1.0, name='a')  # <tf.Variable 'a:0' shape=() dtype=float32_ref>
b = tf.Variable(2.0, name='a')  # <tf.Variable 'a_1:0' shape=() dtype=float32_ref>
c = tf.add(a, b)

# 输入的样本
X = tf.placeholder(tf.float64)
y = tf.placeholder(tf.int32)

with tf.Session() as sess:
    print(sess.run(X, feed_dict={X: [1, 2, 3]}))  # [187.231]
