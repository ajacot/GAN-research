import numpy
import tensorflow as tf

import network
import Fisher
import linalg

d = 2
#d1 = 500 #1000
d2 = 1

d1 = tf.placeholder(tf.int32)
X = tf.placeholder(tf.float32, [None, d])
Y = tf.placeholder(tf.float32, [None, 1])

def shift_relu(x):
    return (tf.nn.relu(x + 0.3) - 0.7)

batch_size = tf.shape(X)[0]

X0 = tf.zeros([1, d]) + 1.0

[YY, Y0], theta = network.affine_net([X, X0], [d, d1, d1, d1, d2], "NET", False, mult_b=1.0)
cost = tf.reduce_mean(tf.square(YY - Y))
'''
d0 = tf.gradients(Y0, theta)
[K0] = Fisher.fwd_gradients([YY], theta, d0)
'''

opt = tf.train.GradientDescentOptimizer(1.0).minimize(cost, var_list=theta)

batch_size = 100
num = 200
num_sizes = 3
num_tries = 1

#kernels = numpy.zeros([batch_size, num])
d_low = numpy.zeros([num, num_tries, num_sizes])
d_high = numpy.zeros([num, num_tries, num_sizes])
d_others = numpy.zeros([num, num_tries, num_sizes])


th = numpy.linspace(0.0, 2.0*numpy.pi, batch_size)
x = numpy.stack([numpy.sin(th), numpy.cos(th)], 1)
const_x = x
low = numpy.reshape(numpy.sin(th*3), [-1, 1]) * numpy.sqrt(2.0)
high = numpy.reshape(numpy.sin(th*5), [-1, 1]) * numpy.sqrt(2.0)

grads_low = tf.gradients(YY, theta, tf.constant(low, tf.float32))
grads_high = tf.gradients(YY, theta, tf.constant(high, tf.float32))

import matplotlib.pyplot as P

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={d1:10000})
    aff_low, aff_high = sess.run([sum([tf.reduce_sum(tf.square(g)) for g in grads_low]),
                                  sum([tf.reduce_sum(tf.square(g)) for g in grads_high])], feed_dict={X:x, d1:10000})

    for i in range(num_sizes):
        dd1 = [100, 1000, 10000][i]
        for j in range(num_tries):
            sess.run(tf.global_variables_initializer(), feed_dict={d1:dd1})
            
            '''
            x = numpy.random.normal(numpy.zeros([batch_size, 1]))
            const_x = numpy.reshape(numpy.linspace(-2.0, 2.0, batch_size), [-1, 1])
            y = numpy.square(x) - 0.5
            '''      

            fx = sess.run(YY, feed_dict={X:x, d1:dd1})
            y = fx + low + high

            for t in range(num):
                dist = sess.run(YY, feed_dict={X:x, d1:dd1}) - y
                d_low[t, j, i] = numpy.mean(dist * low)
                d_high[t, j, i] = numpy.mean(dist * high)
                perp_dist = dist - d_low[t, j, i] * low - d_high[t, j, i] * high
                d_others[t, j, i] = numpy.linalg.norm(perp_dist)
                #d_others[t, j, i] = numpy.sqrt(numpy.mean(numpy.square(dist)) - d_low[t, j, i]**2 - d_high[t, j, i]**2)
                
                for e in range(10):
                    _, err = sess.run([opt, cost], feed_dict={X:x, Y:y, d1:dd1})
                    print(err)

                #y = x[:, 0:1]

time = numpy.linspace(0, num*10, num, endpoint=False)
conv_low = -numpy.exp(-2.0*time * aff_low / (batch_size**2))
conv_high = -numpy.exp(-2.0*time * aff_high / (batch_size**2))

#P.plot(const_x, kernels)
P.plot(d_low[:, :, 0], d_high[:, :, 0], "r:", alpha=0.5, label='$n=100$')
P.plot(d_low[:, :, 1], d_high[:, :, 1], "g:", alpha=0.5, label='$n=1000$')
P.plot(d_low[:, :, 1], d_high[:, :, 2], "b:", alpha=0.5, label='$n=10000$')
P.plot(conv_low, conv_high, label='$n=\infty$')
P.legend()

P.figure()
P.plot(d_others[:, :, 0], "r:", alpha=0.5, label='$n=100$')
P.plot(d_others[:, :, 1], "g:", alpha=0.5, label='$n=1000$')
P.plot(d_others[:, :, 2], "b:", alpha=0.5, label='$n=1000$')
P.legend()


P.show()
