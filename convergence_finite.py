import numpy
import tensorflow as tf

import network
import Fisher
import linalg

# 4000 : 1.62 - 1.66    --- 0.0007
# 2000 : 1.68 - 1.76    --- 0.002
# 1000 : 1.32 - 1.99    --- 0.004

d = 2
#d1 = 500 #1000
d2 = 1

d1 = tf.placeholder(tf.int32)
X = tf.placeholder(tf.float32, [None, d])
Y = tf.placeholder(tf.float32, [None, 1])

batch_size = tf.shape(X)[0]

#X0 = tf.zeros([1, d]) + 1.0
X0 = tf.placeholder(tf.float32, [1, d])

[YY, Y0], theta = network.affine_net([X, X0], [d, d1, d1, d1, d2], "NET",
                                     False, non_lin=tf.nn.relu, mult_b=0.1)
cost = 0.5*tf.reduce_mean(tf.square(YY - Y))

d0 = tf.gradients(Y0, theta)
[K0] = Fisher.fwd_gradients([YY], theta, d0)

dd0 = tf.gradients(Y0, theta[-2:-1])
[C0] = Fisher.fwd_gradients([YY], theta[-2:-1], dd0)

opt = tf.train.GradientDescentOptimizer(1.0).minimize(cost, var_list=theta)
norm_L2 = network.fix_weights(theta)


calc_D = Fisher.derivative([YY], theta)

batch_size = 200
T = 1000
num_sizes = 2
num_tries = 10

points = [30, 70, 130, 170]
goal = [-0.4, -0.2, 0.3, 0.3]
#kernels = numpy.zeros([batch_size, num])
curves = numpy.zeros([batch_size, num_sizes, num_tries])
kernels = numpy.zeros([batch_size, 4])
covars = numpy.zeros([batch_size, 4])

import matplotlib.pyplot as P

with tf.Session() as sess:
    #const_x = numpy.reshape(numpy.linspace(-1.5, 1.5, batch_size), [-1, 1])
    theta = numpy.linspace(-numpy.pi, numpy.pi, batch_size)
    const_x = numpy.stack([numpy.cos(theta), numpy.sin(theta)], 1)

    x = const_x[points, :]
    num_avg = 1
    for i in range(num_avg):
        dd1 = 10000
        sess.run(tf.global_variables_initializer(), feed_dict={d1:dd1})
        
        kernels[:, 0] += sess.run(K0, feed_dict={X:const_x, X0:x[0:1, :], d1:dd1})[:, 0] / num_avg
        kernels[:, 1] += sess.run(K0, feed_dict={X:const_x, X0:x[1:2, :], d1:dd1})[:, 0] / num_avg
        kernels[:, 2] += sess.run(K0, feed_dict={X:const_x, X0:x[2:3, :], d1:dd1})[:, 0] / num_avg
        kernels[:, 3] += sess.run(K0, feed_dict={X:const_x, X0:x[3:4, :], d1:dd1})[:, 0] / num_avg

        covars[:, 0] += sess.run(C0, feed_dict={X:const_x, X0:x[0:1, :], d1:dd1})[:, 0] / num_avg
        covars[:, 1] += sess.run(C0, feed_dict={X:const_x, X0:x[1:2, :], d1:dd1})[:, 0] / num_avg
        covars[:, 2] += sess.run(C0, feed_dict={X:const_x, X0:x[2:3, :], d1:dd1})[:, 0] / num_avg
        covars[:, 3] += sess.run(C0, feed_dict={X:const_x, X0:x[3:4, :], d1:dd1})[:, 0] / num_avg
    
    for i in range(num_sizes):
        dd1 = [50, 1000][i]
        for j in range(num_tries):
            sess.run(tf.global_variables_initializer(), feed_dict={d1:dd1})
            
            
            #fx = sess.run(YY, feed_dict={X:const_x, d1:dd1})
            y = numpy.reshape(goal, [-1, 1])# + fx[points, :]

            for t in range(T):
                _, err = sess.run([opt, cost], feed_dict={X:x, Y:y, d1:dd1})
                sess.run(norm_L2)
                print(err)

            curves[:, i, j] = numpy.reshape(sess.run(YY, feed_dict={X:const_x, d1:dd1}), [-1])
    
K_inv = numpy.linalg.inv(kernels[points, :])
#coeff = numpy.linalg.solve(kernels[points, :], goal)
solve = numpy.dot(kernels, K_inv)
mean = numpy.dot(solve, goal)
variance = numpy.interp(theta+numpy.pi, theta[points]+numpy.pi, numpy.diag(covars[points, :]), period=numpy.pi*2) + numpy.sum((numpy.dot(solve, covars[points, :])-2*covars) * solve, 1)
percentile = numpy.sqrt(numpy.maximum(variance, 0)) * 1.28

#P.plot(const_x, kernels)

P.plot(theta, curves[:, 0, 0], "g:", alpha=0.6, label="$n=50$")
P.plot(theta, curves[:, 0, 1:], "g:", alpha=0.6)
P.plot(theta, curves[:, 1, 0], "r:", alpha=0.6, label="$n=1000$")
P.plot(theta, curves[:, 1, 1:], "r:", alpha=0.6)
P.plot(theta, mean, "b", label="$n=\infty$");
P.plot(theta, mean+percentile, "b:");
P.plot(theta, mean-percentile, "b:");
P.legend()
P.xlabel("$\gamma$")
P.ylabel("$f_{\\theta}(sin(\gamma), cos(\gamma))$")
P.xlim([-numpy.pi, numpy.pi])
P.show()
