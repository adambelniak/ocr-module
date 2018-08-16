from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


x = tf.constant([[1, 2], [2, 1], [2,2], [0, 0], [1, 1]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3], [2]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=10, activation=tf.nn.relu)
linear_model_y = tf.layers.Dense(units=1, )

z = linear_model(x)
y_pred = linear_model_y(z)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
for i in range(3000):
  _, loss_value = sess.run((train, loss))
  print(loss_value)
print(sess.run(y_pred))


