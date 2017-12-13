import numpy
import tensorflow as tf

import network
import Fisher
import linalg

d = 1
X = tf.placeholder(tf.float32, [None, d])
Y = tf.placeholder(tf.float32, [None, 1])

[YY], theta = network.affine_net([X], [d, d+25, d+25, 1], "NET", False)
cost = tf.reduce_mean(tf.square(YY - Y))

F = Fisher.linear_Fisher([YY], theta, a=0.0001)

theta_shapes = [tf.shape(v) for v in theta]
num_eigs = 6
eigs_F, eig_vecs_F, step_eig = linalg.keep_eigs(F, theta_shapes, num_eigs, 2)

dx_F = [Fisher.fwd_gradients([YY], theta, ev)[0] for ev in eig_vecs_F]

dthetaW0 = Fisher.fwd_gradients([YY], [theta[0]], [tf.random_normal(theta[0].shape)])[0]
dthetaW1 = Fisher.fwd_gradients([YY], [theta[2]], [tf.random_normal(theta[2].shape)])[0]
dthetaW2 = Fisher.fwd_gradients([YY], [theta[4]], [tf.random_normal(theta[4].shape)])[0]

dthetab0 = Fisher.fwd_gradients([YY], [theta[1]], [tf.random_normal(theta[1].shape)])[0]
dthetab1 = Fisher.fwd_gradients([YY], [theta[3]], [tf.random_normal(theta[3].shape)])[0]
dthetab2 = Fisher.fwd_gradients([YY], [theta[5]], [tf.random_normal(theta[5].shape)])[0]

opt = tf.train.GradientDescentOptimizer(0.4).minimize(cost, var_list=theta)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 128*4
    x = numpy.random.normal(numpy.zeros([batch_size, d]), 0.3) * [2, 0.5] # + numpy.random.randint(2, size=[batch_size, 1])*2.0 - 1.0
    y = numpy.square(x) - 0.5
    '''
    for _ in range(100):
        err, _ = sess.run([cost, opt], feed_dict={X:x, Y:y})
        print(err)
'''
    for _ in range(30):
        sess.run(step_eig, feed_dict={X:x})

    eigs = sess.run(eigs_F, feed_dict={X:x})
    dxs = sess.run(dx_F, feed_dict={X:x})
    dW0, dW1, dW2, db0, db1, db2 = sess.run([dthetaW0, dthetaW1, dthetaW2, dthetab0, dthetab1, dthetab2],
                                            feed_dict={X:x})


import matplotlib.pyplot as P

for i in range(num_eigs):
    P.scatter(x[:, 0] + i*4, x[:, 1], c=dxs[i][:, 0] / (eigs[i] + 0.0001))

P.figure()

f, axes = P.subplots(2, 3, sharex='col', sharey='row')
axes[0, 0].scatter(x[:, 0], x[:, 1], c=dW0[:, 0])
axes[1, 0].scatter(x[:, 0], x[:, 1], c=db0[:, 0])
axes[0, 1].scatter(x[:, 0], x[:, 1], c=dW1[:, 0])
axes[1, 1].scatter(x[:, 0], x[:, 1], c=db1[:, 0])
axes[0, 2].scatter(x[:, 0], x[:, 1], c=dW2[:, 0])
axes[1, 2].scatter(x[:, 0], x[:, 1], c=db2[:, 0])


P.show()
