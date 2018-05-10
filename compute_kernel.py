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

X0 = tf.placeholder(tf.float32, [1, d])

[YY, Y0], theta = network.affine_net([X, X0], [d, d1, d1, d1, d2], "NET", False
                                     , tf.nn.relu, mult_b=0.1)
cost = tf.reduce_mean(tf.square(YY - Y))

d0 = tf.gradients(Y0, theta)
[K0] = Fisher.fwd_gradients([YY], theta, d0)

opt = tf.train.GradientDescentOptimizer(1.0).minimize(cost, var_list=theta)
norm_L2 = network.fix_weights(theta)

#calc_D = Fisher.derivative([YY], theta)

batch_size = 100
num = 100
num_sizes = 2
num_tries = 5

kernels_init = numpy.zeros([batch_size, num_tries, num_sizes])
kernels_train = numpy.zeros([batch_size, num_tries, num_sizes])

import matplotlib.pyplot as P

with tf.Session() as sess:
    '''
    const_x = numpy.linspace(-1.5, 1.5, batch_size)
    const_x = numpy.reshape(const_x, [-1, 1])
    #grid_x, grid_y = numpy.meshgrid(x, x)
    #x = numpy.stack([numpy.reshape(grid_x, [-1]), numpy.reshape(grid_y, [-1])], 1)
    
    '''
    th = numpy.linspace(-numpy.pi, numpy.pi, batch_size)
    const_x = numpy.stack([numpy.cos(th), numpy.sin(th)], 1)
    
    #sess.run(tf.global_variables_initializer(), feed_dict={d1:1000})
    #kernel = sess.run(K0, feed_dict={X:x, X0:[[1, 0]], d1:1000})

    
    for i in range(num_sizes):
        dd1 = [500, 10000][i]
        for j in range(num_tries):
            sess.run(tf.global_variables_initializer(), feed_dict={d1:dd1})  

            kernels_init[:, j:j+1, i] = sess.run(K0, feed_dict={X:const_x, X0:[[1.0, 0.0]], d1:dd1})
            
            for t in range(num):
                x = numpy.random.normal(numpy.zeros([batch_size, d]))
                y = x[:, 0:1] * x[:, 1:2]
                
                _, err = sess.run([opt, cost], feed_dict={X:x, Y:y, d1:dd1})
                sess.run(norm_L2)
                print(err)
                
            kernels_train[:, j:j+1, i] = sess.run(K0, feed_dict={X:const_x, X0:[[1.0, 0.0]], d1:dd1})
            
#P.plot(x, kernel)
'''
P.plot(th, kernel)
P.figure()
P.semilogy(numpy.fft.fft(kernel, axis=0));
'''
'''
P.imshow(numpy.reshape(kernel, [batch_size, batch_size]))
P.contour(numpy.reshape(kernel, [batch_size, batch_size]))
'''

P.plot(th, kernels_init[:, 0, 0], "g-", alpha=0.5, label='$n=500, t=0$')
P.plot(th, kernels_init[:, 1:, 0], "g-", alpha=0.5)
P.plot(th, kernels_train[:, 0, 0], "g:", alpha=0.5, label='$n=500, t=20$')
P.plot(th, kernels_train[:, 1:, 0], "g:", alpha=0.5)

P.plot(th, kernels_init[:, 0, 1], "r-", alpha=0.5, label='$n=10000, t=0$')
P.plot(th, kernels_init[:, 1:, 1], "r-", alpha=0.5)
P.plot(th, kernels_train[:, 0, 1], "r:", alpha=0.5, label='$n=10000, t=20$')
P.plot(th, kernels_train[:, 1:, 1], "r:", alpha=0.5)


P.xlabel("$\gamma$")
P.xlim([-numpy.pi, numpy.pi])
P.legend()
P.show()
